import functions_framework
import google.cloud.firestore
import datetime
import json
import math

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
    required_fields = ['deviceId', 'timestamp', 'kwhConsumed', 'currentAmp', 'voltageVolt', 'powerWatt', 'dataType']
    for field in required_fields:
        if field not in request_json:
            print(f'Missing required field: {field} in request body: {request_json}')
            return (f'Missing required data field: {field}.', 400)

    # Extract data from the request
    device_id = request_json['deviceId']
    timestamp_mpy_int = request_json['timestamp']
    kwh_consumed = request_json['kwhConsumed']
    current_amp = request_json['currentAmp']
    voltage_volt = request_json['voltageVolt']
    power_watt = request_json['powerWatt']
    energy_source = request_json.get('energySource', 'Grid')
    data_type = request_json['dataType']

    try:
        # --- CRUCIAL TIMESTAMP CONVERSION ---
        # MicroPython's time.time() epoch is Jan 1, 2000, 00:00:00 UTC
        # Python's datetime.fromtimestamp() epoch is Jan 1, 1970, 00:00:00 UTC
        EPOCH_OFFSET_SECONDS_1970_TO_2000 = 946684800
        unix_timestamp_seconds = timestamp_mpy_int + EPOCH_OFFSET_SECONDS_1970_TO_2000
        timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)
        # --- END TIMESTAMP CONVERSION ---

        doc_id = timestamp_dt.isoformat(timespec='seconds').replace('+00:00', 'Z').replace(':', '-')

        if data_type == 'ApplianceEvent':
            # --- Handle Appliance Event Data ---
            if 'applianceSignature' not in request_json:
                return ('Missing applianceSignature data for ApplianceEvent type.', 400)
            
            appliance_signature_data = request_json['applianceSignature']
            
            # Use a separate collection for appliance events
            doc_ref = firestore_client.collection('devices').document(device_id).collection('appliance_signatures').document(doc_id)
            
            data_to_store = {
                'timestamp': timestamp_dt,
                'kwhConsumed': float(kwh_consumed),
                'currentAmp': float(current_amp),
                'voltageVolt': float(voltage_volt),
                'powerWatt': float(power_watt),
                'energySource': energy_source,
                'powerFactor': request_json.get('powerFactor', None),
                'timestamp_esp32_raw': timestamp_mpy_int,
                'signature_data': appliance_signature_data # Store the list of readings here
            }
            doc_ref.set(data_to_store)
            print(f'Appliance signature stored for device: {device_id} at timestamp: {timestamp_dt.isoformat()}')

        elif data_type == 'RegularReading':
            # --- Handle Regular Reading Data ---
            doc_ref = firestore_client.collection('devices').document(device_id).collection('realtime_readings').document(doc_id)

            data_to_store = {
                'timestamp': timestamp_dt,
                'kwhConsumed': float(kwh_consumed),
                'currentAmp': float(current_amp),
                'voltageVolt': float(voltage_volt),
                'powerWatt': float(power_watt),
                'energySource': energy_source,
                'powerFactor': request_json.get('powerFactor', None),
                'timestamp_esp32_raw': timestamp_mpy_int
            }
            doc_ref.set(data_to_store)
            print(f'Regular reading stored for device: {device_id} at timestamp: {timestamp_dt.isoformat()}')
        
        else:
            return ('Invalid dataType provided.', 400)

        return ('Data received successfully!', 200)

    except (ValueError, TypeError) as ve:
        print(f'Data conversion or format error: {ve}')
        return (f'Invalid data format: {ve}', 400)
    except Exception as e:
        print(f'Error writing document to Firestore: {e}')
        return (f'Error processing request and saving data: {e}', 500)
