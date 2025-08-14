import functions_framework
import google.cloud.firestore
import datetime
import pandas as pd
# import lightgbm as lgb
import joblib # Import the joblib library
from google.cloud import storage
import math
import os

# Define the bucket and model file location
MODEL_BUCKET_NAME = 'energreen-prediction-model'
MODEL_FILE_NAME = 'lightgbm_power_prediction_model_energreen_v2.joblib'

# Initialize Firestore and GCS clients
# Initialize clients in the global scope for better performance
firestore_client = google.cloud.firestore.Client()
storage_client = storage.Client()


def get_historical_data(device_id, days=1):
    """
    Fetches historical data from Firestore for a specific device.

    Args:
        device_id (str): The ID of the device.
        days (int): The number of days of historical data to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the historical data.
    """
    end_time = datetime.datetime.now(tz=datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=days)

    # Reference to the realtime_readings collection for the device
    collection_ref = firestore_client.collection('devices').document(device_id).collection('realtime_readings')

    # Query the data within the specified time range
    # Note: Firestore queries are limited to what can be indexed.
    # The 'timestamp' field is a Timestamp object, which is perfect for this.
    query = collection_ref.where('timestamp', '>=', start_time).where('timestamp', '<', end_time).order_by('timestamp')

    docs = query.stream()
    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        data.append({
            'timestamp': doc_data.get('timestamp').isoformat(),
            'powerWatt': doc_data.get('powerWatt'),
            'voltageVolt': doc_data.get('voltageVolt'),
            'currentAmp': doc_data.get('currentAmp'),
            'powerFactor': doc_data.get('powerFactor'),
        })
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    return df


def calculate_lags_and_rolling_means(df):
    """
    Calculates lagged and rolling mean features for the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a 'powerWatt' column and a datetime index.

    Returns:
        pd.DataFrame: The DataFrame with the new features added.
    """
    # Assuming data is at a regular interval (e.g., 1 minute)
    # Lag 1h (60 minutes)
    df['Global_active_power_lag1h'] = df['powerWatt'].shift(60) 
    
    # Lag 24h (1440 minutes)
    df['Global_active_power_lag24h'] = df['powerWatt'].shift(1440) 
    
    # Rolling mean 24h (1440 minutes)
    df['Global_active_power_rolling_mean_24h'] = df['powerWatt'].rolling(window=1440).mean()

    return df


def transform_device_data_for_prediction(raw_data, historical_df):
    """
    Transforms raw device data, including historical features, for the model.
    This is a modified version of your original function.

    Args:
        raw_data (dict): A dictionary containing the raw data.
        historical_df (pd.DataFrame): DataFrame with historical data to calculate features.

    Returns:
        pd.DataFrame: A DataFrame with the correct feature names and calculated values.
    """
    model_features = {}
    
    # Extract date and time features from the timestamp
    timestamp = raw_data.get('timestamp')
    if timestamp is not None:
        # A simple helper function to convert the timestamp
        def to_dt_object(ts):
            EPOCH_OFFSET_SECONDS_1970_TO_2000 = 946684800
            unix_timestamp_seconds = ts + EPOCH_OFFSET_SECONDS_1970_TO_2000
            return datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)
        
        dt_object = to_dt_object(timestamp)
        
        model_features['Year'] = dt_object.year
        model_features['Month'] = dt_object.month
        model_features['Day'] = dt_object.day
        model_features['Hour'] = dt_object.hour
        model_features['DayOfWeek'] = dt_object.weekday()
        model_features['DayOfYear'] = dt_object.timetuple().tm_yday
        model_features['WeekOfYear'] = dt_object.isocalendar()[1]
        model_features['Quarter'] = (dt_object.month - 1) // 3 + 1
        model_features['IsWeekend'] = 1 if dt_object.weekday() >= 5 else 0
        
        month = dt_object.month
        if month in [12, 1, 2]: model_features['Season'] = 1
        elif month in [3, 4, 5]: model_features['Season'] = 2
        elif month in [6, 7, 8]: model_features['Season'] = 3
        else: model_features['Season'] = 4
    
    # Direct mapping and calculation of other features
    model_features['Global_active_power'] = raw_data.get('powerWatt')
    model_features['Voltage'] = raw_data.get('voltageVolt')
    model_features['Global_intensity'] = raw_data.get('currentAmp')
    
    power_factor = raw_data.get('powerFactor')
    power_watt = raw_data.get('powerWatt')
    if power_factor is not None and power_watt is not None and power_factor != 0:
        try:
            pf_clamped = max(-1.0, min(1.0, power_factor))
            angle = math.acos(pf_clamped)
            reactive_power = power_watt * math.tan(angle)
            model_features['Global_reactive_power'] = reactive_power
        except ValueError:
            print("Error calculating reactive power.")
            model_features['Global_reactive_power'] = None
    else:
        model_features['Global_reactive_power'] = None
    
    # Now, add the features that require historical data from the DataFrame
    if not historical_df.empty:
        # Get the latest data point from the historical DataFrame
        latest_data = historical_df.iloc[-1]
        
        # Calculate lag features based on the latest historical data
        model_features['Global_active_power_lag1h'] = latest_data.get('powerWatt')
        model_features['Global_active_power_lag24h'] = latest_data.get('powerWatt')
        
        # Calculate rolling mean
        model_features['Global_active_power_rolling_mean_24h'] = historical_df['powerWatt'].mean()

    # Create a DataFrame from the single transformed data point
    return pd.DataFrame([model_features])


@functions_framework.http
def daily_prediction_runner(request):
    """
    HTTP Cloud Function to run a prediction for a single device.
    It expects a JSON payload with a 'deviceId' key.
    """
    try:
        request_json = request.get_json(silent=True)
        if not request_json or 'deviceId' not in request_json:
            return ('Missing "deviceId" in request body.', 400)
            
        device_id = request_json['deviceId']
        print(f"Starting prediction for device: {device_id}")

        # 1. Download the retrained model from Google Cloud Storage
        bucket = storage_client.bucket(MODEL_BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE_NAME)
        
        # Create a temporary file to save the model
        temp_model_path = '/tmp/model.joblib'
        blob.download_to_filename(temp_model_path)
        
        # 2. Load the model using joblib
        model = joblib.load(temp_model_path)
        print("Model loaded successfully from GCS.")
        
        # 3. Get historical data for the last 24 hours (for lagged features)
        historical_data = get_historical_data(device_id, days=1)
        
        # Check if we have enough data to predict
        if historical_data.empty:
            print(f"Not enough historical data for device {device_id}. Skipping prediction.")
            return (f"Not enough historical data for device {device_id}. Prediction skipped.", 200)

        # 4. Get the latest data point to use for prediction
        latest_row_series = historical_data.iloc[-1]
        raw_data_for_prediction = latest_row_series.to_dict()
        # Correctly get the timestamp from the DataFrame's index
        raw_data_for_prediction['timestamp'] = historical_data.index[-1].timestamp()
        
        # 5. Transform the data into the format the model expects
        features_df = transform_device_data_for_prediction(raw_data_for_prediction, historical_data)

        # Define the expected feature order for the model
        model_feature_names = [
            'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 
            'Quarter', 'IsWeekend', 'Season', 'Global_active_power', 'Global_reactive_power',
            'Voltage', 'Global_intensity', 'Global_active_power_lag1h',
            'Global_active_power_lag24h', 'Global_active_power_rolling_mean_24h'
        ]
        
        # Ensure the feature order matches the model's training order
        features_df = features_df[model_feature_names]

        # 6. Make the prediction
        prediction = model.predict(features_df)
        
        # 7. Store the prediction result in a new Firestore document
        prediction_doc_ref = firestore_client.collection('devices').document(device_id).collection('predictions').document(
            datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        )
        
        prediction_data = {
            'prediction_value': float(prediction[0]), # Convert to standard Python float
            'timestamp': datetime.datetime.now(tz=datetime.timezone.utc),
            'prediction_date': datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()
        }
        prediction_doc_ref.set(prediction_data)

        print(f"Prediction for device {device_id} successful. Result: {prediction_data}")

        return (f'Prediction job completed successfully for device: {device_id}', 200)

    except Exception as e:
        print(f"An error occurred: {e}")
        return (f"Prediction failed with an error: {e}", 500)