import functions_framework
import google.cloud.firestore
import datetime
import json
import math

def transform_device_data(raw_data):
    """
    Transforms raw device data into a format suitable for the LightGBM model.

    This function maps the keys from the raw ESP32 data to the feature names
    expected by the trained LightGBM model and calculates additional features.

    Args:
        raw_data (dict): A dictionary containing the raw data from the ESP32.

    Returns:
        dict: A dictionary with the correct feature names and calculated values.
    """
    model_features = {
        'Global_active_power': None,
        'Global_reactive_power': None,
        'Voltage': None,
        'Global_intensity': None,
        'Year': None,
        'Month': None,
        'Day': None,
        'Hour': None,
        'DayOfWeek': None,
        'DayOfYear': None,
        'WeekOfYear': None,
        'Quarter': None,
        'Season': None,
        'IsWeekend': None,
        'Global_active_power_lag1h': None,
        'Global_active_power_lag24h': None,
        'Global_active_power_rolling_mean_24h': None,
    }

    # Direct mapping of available data fields
    model_features['Global_active_power'] = raw_data.get('powerWatt')
    model_features['Voltage'] = raw_data.get('voltageVolt')
    model_features['Global_intensity'] = raw_data.get('currentAmp')

    # Calculate Global_reactive_power from available data
    power_factor = raw_data.get('powerFactor')
    power_watt = raw_data.get('powerWatt')

    if power_factor is not None and power_watt is not None and power_factor != 0:
        try:
            # We use math.acos() which requires a value between -1 and 1.
            # Clamp the value to handle potential floating point errors.
            pf_clamped = max(-1.0, min(1.0, power_factor))
            angle = math.acos(pf_clamped)
            # Reactive power (Q) = Active Power (P) * tan(acos(Power Factor))
            reactive_power = power_watt * math.tan(angle)
            model_features['Global_reactive_power'] = reactive_power
        except ValueError:
            print("Error calculating reactive power: Invalid power factor value.")
            model_features['Global_reactive_power'] = None

    # Extract date and time features from the timestamp
    timestamp = raw_data.get('timestamp')
    if timestamp is not None:
        # Assuming the timestamp has been corrected to the Unix epoch
        # in the main Cloud Function, we can use it directly.
        # This will need to be passed from the main function.
        # For now, let's assume the raw timestamp is the corrected one.
        EPOCH_OFFSET_SECONDS_1970_TO_2000 = 946684800
        unix_timestamp_seconds = timestamp + EPOCH_OFFSET_SECONDS_1970_TO_2000
        dt_object = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)

        model_features['Year'] = dt_object.year
        model_features['Month'] = dt_object.month
        model_features['Day'] = dt_object.day
        model_features['Hour'] = dt_object.hour
        model_features['DayOfWeek'] = dt_object.weekday()  # 0=Monday, 6=Sunday
        model_features['DayOfYear'] = dt_object.timetuple().tm_yday
        model_features['WeekOfYear'] = dt_object.isocalendar()[1]
        model_features['Quarter'] = (dt_object.month - 1) // 3 + 1
        model_features['IsWeekend'] = 1 if dt_object.weekday() >= 5 else 0

        # Determine the Season (approximate)
        month = dt_object.month
        if month in [12, 1, 2]:
            model_features['Season'] = 1  # Winter
        elif month in [3, 4, 5]:
            model_features['Season'] = 2  # Spring
        elif month in [6, 7, 8]:
            model_features['Season'] = 3  # Summer
        else:
            model_features['Season'] = 4  # Autumn

    # Note: The lagged and rolling mean features must be computed
    # by querying historical data from Firestore. This function
    # cannot calculate them from a single data point.

    return model_features


@functions_framework.http
def receive_energy_data(request):
    """
    HTTP Cloud Function to receive energy data from IoT devices and store it in Firestore.
    Expects a POST request with a JSON body containing energy readings.
    """
    firestore_client = google.cloud.firestore.Client()

    if request.method != 'POST':
        return ('Method Not Allowed', 405)

    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return ('Request body must be JSON', 400)
    except Exception as e:
        return (f'Error parsing JSON: {e}', 400)

    # Basic validation for incoming data fields
    required_fields = ['deviceId', 'timestamp', 'kwhConsumed', 'currentAmp', 'voltageVolt', 'powerWatt']
    for field in required_fields:
        if field not in request_json:
            print(f'Missing required field: {field} in request body: {request_json}')
            return (f'Missing required data field: {field}.', 400)

    # Extract data from the request
    device_id = request_json['deviceId']
    # 'timestamp' from ESP32 is an integer (seconds since 2000-01-01 UTC)
    timestamp_mpy_int = request_json['timestamp']
    kwh_consumed = request_json['kwhConsumed']
    current_amp = request_json['currentAmp']
    voltage_volt = request_json['voltageVolt']
    power_watt = request_json['powerWatt']
    energy_source = request_json.get('energySource', 'Grid')  # Default to 'Grid' if not provided

    try:
        # --- CRUCIAL TIMESTAMP CONVERSION ---
        # MicroPython's time.time() epoch is Jan 1, 2000, 00:00:00 UTC
        # Python's datetime.fromtimestamp() epoch is Jan 1, 1970, 00:00:00 UTC
        # Calculate the offset in seconds between these two epochs
        EPOCH_OFFSET_SECONDS_1970_TO_2000 = 946684800  # (30 years * 365.25 days/year * 24 hours * 60 min * 60 sec)

        # Convert MicroPython timestamp to Unix timestamp (seconds since 1970)
        unix_timestamp_seconds = timestamp_mpy_int + EPOCH_OFFSET_SECONDS_1970_TO_2000

        # Convert Unix timestamp to a Python datetime object (in UTC)
        timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)
        # --- END TIMESTAMP CONVERSION ---

        # Format datetime object for use as a Firestore Document ID
        # Example format: '2025-07-04T01:45:00Z' (from your screenshot)
        # We replace colons with hyphens as they can sometimes cause issues in document IDs depending on context
        doc_id = timestamp_dt.isoformat(timespec='seconds').replace('+00:00', 'Z').replace(':', '-')  # Ensures 'Z' for UTC and clean ID

        # Reference to the specific device's realtime_readings subcollection, using formatted timestamp as ID
        doc_ref = firestore_client.collection('devices').document(device_id).collection('realtime_readings').document(doc_id)

        # Prepare data for Firestore
        data_to_store = {
            'timestamp': timestamp_dt,  # Store as datetime object, Firestore auto-converts to native Timestamp
            'kwhConsumed': float(kwh_consumed),
            'currentAmp': float(current_amp),
            'voltageVolt': float(voltage_volt),
            'powerWatt': float(power_watt),
            'energySource': energy_source,
            'powerFactor': request_json.get('powerFactor', None), # Include power factor
            # Optional: store the raw MicroPython timestamp for debugging/auditing
            'timestamp_esp32_raw': timestamp_mpy_int
        }

        # Store data in Firestore
        doc_ref.set(data_to_store)

        print(f'Data received and stored for device: {device_id} at timestamp: {timestamp_dt.isoformat()}')
        return ('Data received successfully!', 200)

    except (ValueError, TypeError) as ve:  # Catch type/value errors from conversion
        print(f'Data conversion or format error: {ve}')
        return (f'Invalid data format: {ve}', 400)
    except Exception as e:
        print(f'Error writing document to Firestore: {e}')
        return (f'Error processing request and saving data: {e}', 500)


# --- Example of how to use the new function (do not add this to your main.py file) ---
# This part is for your reference only, to show how you would call the new function
# and get your data ready for a prediction. You would need to load your model here.
#
# if __name__ == '__main__':
#    sample_raw_data = {
#        "powerFactor": 0.86,
#        "kwhConsumed": 244.11,
#        "energySource": "Grid",
#        "timestamp": 808487401, # This is the raw MicroPython timestamp
#        "frequencyHz": 60.0,
#        "currentAmp": 3.788,
#        "deviceId": "energreen_esp32_001",
#        "voltageVolt": 224.2,
#        "powerWatt": 695.5
#    }
#
#    # Call the new transformation function to prepare the data
#    transformed_data = transform_device_data(sample_raw_data)
#
#    # Here, you would load your pre-trained LightGBM model.
#    # import lightgbm as lgb
#    # model = lgb.Booster(model_file='your_model.txt')
#
#    # Prepare the data for prediction (e.g., convert to a list of values)
#    # Ensure the order of features matches the training data.
#    # features_for_prediction = [transformed_data[feature] for feature in list(transformed_data.keys())]
#
#    # Make a prediction
#    # prediction = model.predict([features_for_prediction])
#    # print("Prediction:", prediction)
