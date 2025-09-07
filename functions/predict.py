import functions_framework
import google.cloud.firestore
import datetime
import pandas as pd
import joblib
from google.cloud import storage
import math
import os

# Define the bucket and model file location
MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET")
MODEL_FILE_NAME = 'lightgbm_power_prediction_model_energreen_v2.joblib'

# Initialize Firestore and GCS clients
# Initialize clients in the global scope for better performance
firestore_client = google.cloud.firestore.Client()
storage_client = storage.Client()

def get_historical_data(device_id, days=2):
    """
    Fetches historical data from Firestore for a specific device.

    Args:
        device_id (str): The ID of the device.
        days (int): The number of days of historical data to fetch.
                    We fetch 2 days to ensure we have enough data for 24h lags.

    Returns:
        pd.DataFrame: A DataFrame containing the historical data.
    """
    end_time = datetime.datetime.now(tz=datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=days)

    collection_ref = firestore_client.collection('devices').document(device_id).collection('realtime_readings')
    query = collection_ref.where('timestamp', '>=', start_time).where('timestamp', '<', end_time).order_by('timestamp')

    docs = query.stream()
    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        data.append({
            'timestamp': doc_data.get('timestamp'),
            'powerWatt': doc_data.get('powerWatt'),
            'voltageVolt': doc_data.get('voltageVolt'),
            'currentAmp': doc_data.get('currentAmp'),
            'powerFactor': doc_data.get('powerFactor'),
        })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp').sort_index()

    return df


def create_prediction_features(historical_df, target_datetime):
    """
    Creates the necessary features (lagged values, rolling mean, etc.) for a
    given target datetime based on the historical data. This version is more
    resilient and only creates features that have enough historical data.

    Args:
        historical_df (pd.DataFrame): DataFrame with historical data.
        target_datetime (datetime.datetime): The specific future datetime to predict for.

    Returns:
        dict: A dictionary of the features, or None if there's no data at all.
    """
    # Get the historical data *before* the target datetime
    data_for_lags = historical_df[historical_df.index < target_datetime]
    
    if data_for_lags.empty:
        print(f"Skipping feature creation for {target_datetime}. No historical data available.")
        return None

    model_features = {}
    
    # --- Date and Time Features ---
    model_features['Year'] = target_datetime.year
    model_features['Month'] = target_datetime.month
    model_features['Day'] = target_datetime.day
    model_features['Hour'] = target_datetime.hour
    model_features['DayOfWeek'] = target_datetime.weekday()
    model_features['DayOfYear'] = target_datetime.timetuple().tm_yday
    model_features['WeekOfYear'] = target_datetime.isocalendar()[1]
    model_features['Quarter'] = (target_datetime.month - 1) // 3 + 1
    model_features['IsWeekend'] = 1 if target_datetime.weekday() >= 5 else 0
    
    month = target_datetime.month
    if month in [12, 1, 2]: model_features['Season'] = 1
    elif month in [3, 4, 5]: model_features['Season'] = 2
    elif month in [6, 7, 8]: model_features['Season'] = 3
    else: model_features['Season'] = 4
    
    # --- Historical Features (Lags & Rolling Mean) ---
    # Use latest available readings for real-time features
    latest_data = data_for_lags.iloc[-1]
    model_features['Voltage'] = latest_data['voltageVolt']
    model_features['Global_intensity'] = latest_data['currentAmp']
    
    # Calculate Global_active_power_lag1h if possible
    timestamp_1h_ago = target_datetime - datetime.timedelta(hours=1)
    lag_1h_data = historical_df[historical_df.index <= timestamp_1h_ago]
    if not lag_1h_data.empty:
        model_features['Global_active_power_lag1h'] = lag_1h_data.iloc[-1]['powerWatt']
    else:
        # Default value if not enough data
        model_features['Global_active_power_lag1h'] = 0.0

    # Calculate Global_active_power_lag24h if possible
    timestamp_24h_ago = target_datetime - datetime.timedelta(hours=24)
    lag_24h_data = historical_df[historical_df.index <= timestamp_24h_ago]
    if not lag_24h_data.empty:
        model_features['Global_active_power_lag24h'] = lag_24h_data.iloc[-1]['powerWatt']
    else:
        # Default value if not enough data
        model_features['Global_active_power_lag24h'] = 0.0
        
    # Calculate rolling mean for the 24 hours if possible
    rolling_mean_data = historical_df[
        (historical_df.index >= target_datetime - datetime.timedelta(hours=24)) &
        (historical_df.index < target_datetime)
    ]['powerWatt']
    if not rolling_mean_data.empty:
        model_features['Global_active_power_rolling_mean_24h'] = rolling_mean_data.mean()
    else:
        # Default value if not enough data
        model_features['Global_active_power_rolling_mean_24h'] = 0.0
        
    # Reactive power calculation based on latest data
    power_factor = latest_data['powerFactor']
    power_watt = latest_data['powerWatt']
    if power_factor is not None and power_watt is not None and power_factor != 0:
        try:
            pf_clamped = max(-1.0, min(1.0, power_factor))
            angle = math.acos(pf_clamped)
            reactive_power = power_watt * math.tan(angle)
            model_features['Global_reactive_power'] = reactive_power
        except ValueError:
            model_features['Global_reactive_power'] = None
    else:
        model_features['Global_reactive_power'] = None

    features_df = pd.DataFrame([model_features])
    
    # Ensure correct data types and feature order
    features_df['Global_reactive_power'] = features_df['Global_reactive_power'].fillna(0).astype(float)
    features_df['Season'] = features_df['Season'].astype('category')
    
    # Define the expected feature order for the model
    model_feature_names = [
        'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 
        'Quarter', 'IsWeekend', 'Season', 'Global_reactive_power',
        'Voltage', 'Global_intensity', 'Global_active_power_lag1h',
        'Global_active_power_lag24h', 'Global_active_power_rolling_mean_24h'
    ]
    
    return features_df[model_feature_names]


@functions_framework.http
def daily_prediction_runner(request):
    """
    HTTP Cloud Function to run a multi-interval prediction for a single device.
    It expects a JSON payload with a 'deviceId' key.
    """
    try:
        request_json = request.get_json(silent=True)
        if not request_json or 'deviceId' not in request_json:
            return ('Missing "deviceId" in request body.', 400)
            
        device_id = request_json['deviceId']
        print(f"Starting multi-interval prediction for device: {device_id}")

        # 1. Download and load the retrained model
        bucket = storage_client.bucket(MODEL_BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE_NAME)
        temp_model_path = '/tmp/model.joblib'
        blob.download_to_filename(temp_model_path)
        model = joblib.load(temp_model_path)
        print("Model loaded successfully from GCS.")
        
        # 2. Get historical data for the last 48 hours to ensure we have enough data
        historical_data = get_historical_data(device_id, days=2)
        
        if historical_data.empty:
            print(f"No historical data found for device {device_id}. Skipping all predictions.")
            return (f"No historical data found for device {device_id}. Prediction skipped.", 200)

        # 3. Define the target datetimes for our multi-interval predictions
        current_time = datetime.datetime.now(tz=datetime.timezone.utc)
        predictions_to_store = []
        
        # --- Prediction for next data point (e.g., next minute) ---
        target_time_immediate = current_time + datetime.timedelta(minutes=1)
        features_immediate = create_prediction_features(historical_data, target_time_immediate)
        if features_immediate is not None:
            prediction_immediate = model.predict(features_immediate)[0]
            predictions_to_store.append({
                'prediction_value': float(prediction_immediate),
                'prediction_for_time': target_time_immediate,
                'interval': 'Immediate'
            })
        
        # --- Prediction for one hour from now ---
        target_time_next_hour = current_time + datetime.timedelta(hours=1)
        features_next_hour = create_prediction_features(historical_data, target_time_next_hour)
        if features_next_hour is not None:
            prediction_next_hour = model.predict(features_next_hour)[0]
            predictions_to_store.append({
                'prediction_value': float(prediction_next_hour),
                'prediction_for_time': target_time_next_hour,
                'interval': 'Next Hour'
            })

        # --- Prediction for the same time tomorrow ---
        target_time_next_day = current_time + datetime.timedelta(days=1)
        features_next_day = create_prediction_features(historical_data, target_time_next_day)
        if features_next_day is not None:
            prediction_next_day = model.predict(features_next_day)[0]
            predictions_to_store.append({
                'prediction_value': float(prediction_next_day),
                'prediction_for_time': target_time_next_day,
                'interval': 'Next Day'
            })

        if not predictions_to_store:
            print(f"Not enough historical data for any prediction for device {device_id}. Skipping.")
            return (f"Not enough historical data for any prediction for device {device_id}. Prediction skipped.", 200)

        # 4. Store all predictions in a single Firestore document
        predictions_batch_doc_ref = firestore_client.collection('devices').document(device_id).collection('predictions').document()
        
        prediction_data = {
            'timestamp': current_time,
            'predictions': predictions_to_store
        }
        predictions_batch_doc_ref.set(prediction_data)

        print(f"Multi-interval prediction for device {device_id} successful. Results stored in Firestore.")

        return (f'Multi-interval prediction job completed successfully for device: {device_id}', 200)

    except Exception as e:
        print(f"An error occurred: {e}")
        return (f"Prediction failed with an error: {e}", 500)
