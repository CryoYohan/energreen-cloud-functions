import functions_framework
import google.cloud.firestore
import datetime
import json

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
    energy_source = request_json.get('energySource', 'Grid') # Default to 'Grid' if not provided

    try:
        # --- CRUCIAL TIMESTAMP CONVERSION ---
        # MicroPython's time.time() epoch is Jan 1, 2000, 00:00:00 UTC
        # Python's datetime.fromtimestamp() epoch is Jan 1, 1970, 00:00:00 UTC
        # Calculate the offset in seconds between these two epochs
        EPOCH_OFFSET_SECONDS_1970_TO_2000 = 946684800 # (30 years * 365.25 days/year * 24 hours * 60 min * 60 sec)

        # Convert MicroPython timestamp to Unix timestamp (seconds since 1970)
        unix_timestamp_seconds = timestamp_mpy_int + EPOCH_OFFSET_SECONDS_1970_TO_2000

        # Convert Unix timestamp to a Python datetime object (in UTC)
        timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)
        # --- END TIMESTAMP CONVERSION ---

        # Format datetime object for use as a Firestore Document ID
        # Example format: '2025-07-04T01:45:00Z' (from your screenshot)
        # We replace colons with hyphens as they can sometimes cause issues in document IDs depending on context
        doc_id = timestamp_dt.isoformat(timespec='seconds').replace('+00:00', 'Z').replace(':', '-') # Ensures 'Z' for UTC and clean ID

        # Reference to the specific device's realtime_readings subcollection, using formatted timestamp as ID
        doc_ref = firestore_client.collection('devices').document(device_id).collection('realtime_readings').document(doc_id)

        # Prepare data for Firestore
        data_to_store = {
            'timestamp': timestamp_dt, # Store as datetime object, Firestore auto-converts to native Timestamp
            'kwhConsumed': float(kwh_consumed),
            'currentAmp': float(current_amp),
            'voltageVolt': float(voltage_volt),
            'powerWatt': float(power_watt),
            'energySource': energy_source,
            # Optional: store the raw MicroPython timestamp for debugging/auditing
            'timestamp_esp32_raw': timestamp_mpy_int
        }

        # Store data in Firestore
        doc_ref.set(data_to_store)

        print(f'Data received and stored for device: {device_id} at timestamp: {timestamp_dt.isoformat()}')
        return ('Data received successfully!', 200)

    except (ValueError, TypeError) as ve: # Catch type/value errors from conversion
        print(f'Data conversion or format error: {ve}')
        return (f'Invalid data format: {ve}', 400)
    except Exception as e:
        print(f'Error writing document to Firestore: {e}')
        return (f'Error processing request and saving data: {e}', 500)