import functions_framework
import functions_framework
import google.cloud.firestore
import datetime
import pandas as pd
import joblib
from google.cloud import storage
import math
import os
import numpy as np
from prophet.serialize import model_from_json

# Define the bucket and model file location
MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET")
LIGHTGBM_FILE_NAME = "forecasting-ai-models/lightgbm_model.joblib"
PROPHET_FILE_NAME = "forecasting-ai-models/prophet_model.joblib"

# Initialize Firestore and GCS clients
firestore_client = google.cloud.firestore.Client()
storage_client = storage.Client()

# --- Utility: Load models ---
def load_lightgbm_model():
    bucket = storage_client.bucket(MODEL_BUCKET_NAME)
    blob = bucket.blob(LIGHTGBM_FILE_NAME)
    temp_path = "/tmp/lightgbm_model.joblib"
    blob.download_to_filename(temp_path)
    return joblib.load(temp_path)

def load_prophet_model():
    bucket = storage_client.bucket(MODEL_BUCKET_NAME)
    blob = bucket.blob(PROPHET_FILE_NAME)
    temp_path = "/tmp/prophet_model.joblib"
    blob.download_to_filename(temp_path)
    return joblib.load(temp_path)

# HELPER FUNCTIONS 
def get_historical_data(device_id, days=2):
    """Fetch historical readings for a device from Firestore."""
    end_time = datetime.datetime.now(tz=datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=days)

    collection_ref = (
        firestore_client.collection('devices')
        .document(device_id)
        .collection('realtime_readings')
    )
    query = (
        collection_ref.where('timestamp', '>=', start_time)
        .where('timestamp', '<', end_time)
        .order_by('timestamp')
    )

    docs = query.stream()
    data = [
        {
            'timestamp': d.get('timestamp'),
            'powerWatt': d.get('powerWatt'),
            'voltageVolt': d.get('voltageVolt'),
            'currentAmp': d.get('currentAmp'),
            'powerFactor': d.get('powerFactor'),
        }
        for doc in docs if (d := doc.to_dict())
    ]

    df = pd.DataFrame(data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp').sort_index()
    return df


def create_prediction_features(historical_df, target_datetime):
    """Builds feature set for prediction at a given datetime."""
    data_for_lags = historical_df[historical_df.index < target_datetime]
    if data_for_lags.empty:
        return None

    model_features = {
        'Year': target_datetime.year,
        'Month': target_datetime.month,
        'Day': target_datetime.day,
        'Hour': target_datetime.hour,
        'DayOfWeek': target_datetime.weekday(),
        'DayOfYear': target_datetime.timetuple().tm_yday,
        'WeekOfYear': target_datetime.isocalendar()[1],
        'Quarter': (target_datetime.month - 1) // 3 + 1,
        'IsWeekend': 1 if target_datetime.weekday() >= 5 else 0,
        'Season': (
            1 if target_datetime.month in [12, 1, 2] else
            2 if target_datetime.month in [3, 4, 5] else
            3 if target_datetime.month in [6, 7, 8] else 4
        ),
    }

    # Latest available readings
    latest = data_for_lags.iloc[-1]
    model_features['Voltage'] = latest['voltageVolt']
    model_features['Global_intensity'] = latest['currentAmp']

    # Lags
    for hours, key in [(1, 'Global_active_power_lag1h'), (24, 'Global_active_power_lag24h')]:
        ts_ago = target_datetime - datetime.timedelta(hours=hours)
        lag_data = historical_df[historical_df.index <= ts_ago]
        model_features[key] = lag_data.iloc[-1]['powerWatt'] if not lag_data.empty else 0.0

    # Rolling mean
    rolling = historical_df[
        (historical_df.index >= target_datetime - datetime.timedelta(hours=24)) &
        (historical_df.index < target_datetime)
    ]['powerWatt']
    model_features['Global_active_power_rolling_mean_24h'] = rolling.mean() if not rolling.empty else 0.0

    # Reactive power
    pf, pw = latest['powerFactor'], latest['powerWatt']
    if pf and pw and pf != 0:
        try:
            angle = math.acos(max(-1.0, min(1.0, pf)))
            model_features['Global_reactive_power'] = pw * math.tan(angle)
        except ValueError:
            model_features['Global_reactive_power'] = 0.0
    else:
        model_features['Global_reactive_power'] = 0.0

    features_df = pd.DataFrame([model_features])
    #features_df['Season'] = features_df['Season'].astype('category')

    feature_order = [
        'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
        'Quarter', 'IsWeekend', 'Season', 'Global_reactive_power',
        'Voltage', 'Global_intensity', 'Global_active_power_lag1h',
        'Global_active_power_lag24h', 'Global_active_power_rolling_mean_24h'
    ]
    return features_df[feature_order]


def calculate_kwh(predicted_watt, hours):
    """Convert predicted watts to kWh over a given time horizon."""
    return round((predicted_watt * hours) / 1000.0, 3)


def estimate_confidence(predictions, base_value, conf_level=0.95):
    """Estimate confidence interval based on residual variability."""
    if len(predictions) < 2:
        # fallback: +/- 10% of base
        margin = 0.1 * base_value
        return base_value - margin, base_value + margin

    std = np.std(predictions)
    margin = 1.96 * std  # 95% CI
    return base_value - margin, base_value + margin


def calculate_accuracy(historical_df, model):
    """Rough model accuracy using recent backtesting."""
    try:
        if historical_df.shape[0] < 48:
            return None  # not enough history

        test_points = historical_df.iloc[-24:]  # last 24 points
        errors = []
        for ts, row in test_points.iterrows():
            feats = create_prediction_features(historical_df, ts)
            if feats is not None:
                pred = model.predict(feats)[0]
                if row['powerWatt']:
                    errors.append(abs((row['powerWatt'] - pred) / row['powerWatt']))

        if errors:
            return round(1 - np.mean(errors), 2)  # ~accuracy %
        return None
    except Exception:
        return None


@functions_framework.http
def daily_prediction_runner(request):
    """HTTP-triggered prediction function for multiple intervals."""
    try:
        request_json = request.get_json(silent=True)
        if not request_json or "deviceId" not in request_json:
            return ('Missing "deviceId" in request body.', 400)

        device_id = request_json["deviceId"]
        print(f"Starting multi-interval prediction for device: {device_id}")

        # Load both models
        lightgbm_model = load_lightgbm_model()
        prophet_model = load_prophet_model()

        # Historical data
        historical = get_historical_data(device_id, days=2)
        if historical.empty:
            return (f"No historical data found for device {device_id}. Prediction skipped.", 200)

        current_time = datetime.datetime.now(tz=datetime.timezone.utc)
        intervals = [
            ("Immediate", current_time + datetime.timedelta(minutes=1), 1/60),
            ("Next Hour", current_time + datetime.timedelta(hours=1), 1),
            ("Next Day", current_time + datetime.timedelta(days=1), 24),
            ("Next Week", current_time + datetime.timedelta(weeks=1), 24 * 7),
            ("Next Month", current_time + datetime.timedelta(days=30), 24 * 30),
        ]

        predictions_to_store = {"lightgbm": [], "prophet": []}
        baseline_consumption = historical["powerWatt"].mean() if not historical.empty else 0

        # --- LightGBM Predictions ---
        for label, target_time, hours in intervals:
            features = create_prediction_features(historical, target_time)
            if features is None:
                continue

            predicted_watt = float(lightgbm_model.predict(features)[0])
            predicted_kwh = calculate_kwh(predicted_watt, hours)
            boot_preds = [float(lightgbm_model.predict(features)[0]) for _ in range(5)]
            lower, upper = estimate_confidence(boot_preds, predicted_watt)

            trend_pct = None
            if baseline_consumption and baseline_consumption > 0:
                trend_pct = round(((predicted_watt - baseline_consumption) / baseline_consumption) * 100, 2)

            predictions_to_store["lightgbm"].append({
                "interval": label,
                "prediction_for_time": target_time,
                "prediction_value_watt": predicted_watt,
                "predicted_consumption_kwh": predicted_kwh,
                "confidence_interval_watt": {"lower": lower, "upper": upper},
                "trend_vs_baseline_percent": trend_pct
            })

        # --- Prophet Predictions (time-series forecast) ---
        future_df = prophet_model.make_future_dataframe(periods=24*30, freq="H")  # up to 1 month hourly
        forecast = prophet_model.predict(future_df)
        forecast = forecast.set_index("ds")

        for label, target_time, hours in intervals:
            if target_time not in forecast.index:
                continue

            row = forecast.loc[target_time]
            predicted_watt = float(row["yhat"])
            predicted_kwh = calculate_kwh(predicted_watt, hours)
            lower, upper = float(row["yhat_lower"]), float(row["yhat_upper"])

            trend_pct = None
            if baseline_consumption and baseline_consumption > 0:
                trend_pct = round(((predicted_watt - baseline_consumption) / baseline_consumption) * 100, 2)

            predictions_to_store["prophet"].append({
                "interval": label,
                "prediction_for_time": target_time,
                "prediction_value_watt": predicted_watt,
                "predicted_consumption_kwh": predicted_kwh,
                "confidence_interval_watt": {"lower": lower, "upper": upper},
                "trend_vs_baseline_percent": trend_pct
            })

        # Save to Firestore
        doc_ref = (
            firestore_client.collection("devices")
            .document(device_id)
            .collection("predictions")
            .document()
        )
        doc_ref.set({
            "timestamp": current_time,
            "predictions": predictions_to_store,
        })

        return (f"Predictions stored for device: {device_id}", 200)

    except Exception as e:
        print(f"Prediction failed: {e}")
        return (f"Prediction failed: {e}", 500)
