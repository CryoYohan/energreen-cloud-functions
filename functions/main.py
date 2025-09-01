import functions_framework
import google.cloud.firestore
import datetime
import json
import math

def predict_appliance_type(signature_data):
    """
    A simple rule-based model to predict appliance type based on average power consumption.
    This can be replaced with a more advanced ML model later.
    """
    if not signature_data:
        return 'Unknown'
    
    # Calculate the average power consumption for the signature
    total_power = sum(float(d.get('powerWatt', 0)) for d in signature_data)
    average_power = total_power / len(signature_data)
    
    # Simple prediction rules
    if average_power > 1500:
        return 'Oven'
    elif average_power > 1000:
        return 'Electric Kettle'
    elif average_power > 500:
        return 'Microwave'
    elif average_power > 100:
        return 'Toaster'
    elif average_power > 50:
        return 'Lightbulb'
    else:
        return 'Standby Power'

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

    # Validate that 'dataType' and 'deviceId' are always present
    if 'dataType' not in request_json or 'deviceId' not in request_json:
        return ('Missing required data fields: dataType or deviceId.', 400)

    data_type = request_json['dataType']
    device_id = request_json['deviceId']

    try:
        # --- CRUCIAL TIMESTAMP CONVERSION ---
        # MicroPython's time.time() epoch is Jan 1, 2000, 00:00:00 UTC
        # Python's datetime.fromtimestamp() epoch is Jan 1, 1970, 00:00:00 UTC
        EPOCH_OFFSET_SECONDS_1970_TO_2000 = 946684800
        
        # Determine the target collection based on the data type
        if data_type == 'ApplianceSignature':
            # Handle signature data
            if 'signature_data' not in request_json:
                return ('Missing required data field: signature_data for ApplianceSignature type.', 400)

            # Use the timestamp from the first reading in the signature for the doc ID
            first_reading_timestamp_mpy = request_json['signature_data'][0]['timestamp']
            unix_timestamp_seconds = first_reading_timestamp_mpy + EPOCH_OFFSET_SECONDS_1970_TO_2000
            timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)
            
            doc_id = timestamp_dt.isoformat(timespec='seconds').replace('+00:00', 'Z').replace(':', '-')

            # 1. Store the raw signature data
            data_to_store = {
                'deviceId': device_id,
                'timestamp': timestamp_dt,
                'signature_data': request_json['signature_data'],
                'processed_at': datetime.datetime.now(tz=datetime.timezone.utc)
            }
            doc_ref = firestore_client.collection('devices').document(device_id).collection('appliance_signatures').document(doc_id)
            doc_ref.set(data_to_store)
            print(f'Appliance signature stored for device: {device_id} at timestamp: {timestamp_dt.isoformat()}')

            # 2. Predict the appliance type and store it separately
            predicted_appliance = predict_appliance_type(request_json['signature_data'])
            prediction_data = {
                'deviceId': device_id,
                'timestamp': timestamp_dt,
                'predictedAppliance': predicted_appliance,
                'processed_at': datetime.datetime.now(tz=datetime.timezone.utc)
            }
            prediction_doc_ref = firestore_client.collection('devices').document(device_id).collection('predicted_appliances').document(doc_id)
            prediction_doc_ref.set(prediction_data)
            print(f'Prediction stored for device: {device_id}. Predicted appliance: {predicted_appliance}')

        elif data_type == 'RegularReading':
            # Handle regular reading data
            required_fields = ['timestamp', 'kwhConsumed', 'currentAmp', 'voltageVolt', 'powerWatt']
            for field in required_fields:
                if field not in request_json:
                    print(f'Missing required field: {field} in request body: {request_json}')
                    return (f'Missing required data field: {field}.', 400)

            timestamp_mpy_int = request_json['timestamp']
            unix_timestamp_seconds = timestamp_mpy_int + EPOCH_OFFSET_SECONDS_1970_TO_2000
            timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)
            doc_id = timestamp_dt.isoformat(timespec='seconds').replace('+00:00', 'Z').replace(':', '-')

            data_to_store = {
                'timestamp': timestamp_dt,
                'kwhConsumed': float(request_json['kwhConsumed']),
                'currentAmp': float(request_json['currentAmp']),
                'voltageVolt': float(request_json['voltageVolt']),
                'powerWatt': float(request_json['powerWatt']),
                'energySource': request_json.get('energySource', 'Grid'),
                'powerFactor': request_json.get('powerFactor', None),
                'timestamp_esp32_raw': timestamp_mpy_int
            }
            doc_ref = firestore_client.collection('devices').document(device_id).collection('realtime_readings').document(doc_id)
            doc_ref.set(data_to_store)
            print(f'Regular reading stored for device: {device_id} at timestamp: {timestamp_dt.isoformat()}')

        else:
            return ('Invalid dataType provided.', 400)

        return ('Data received and processed successfully!', 200)

    except (ValueError, TypeError) as ve:
        print(f'Data conversion or format error: {ve}')
        return (f'Invalid data format: {ve}', 400)
    except Exception as e:
        print(f'Error processing request and saving data: {e}')
        return (f'Error processing request and saving data: {e}', 500)
