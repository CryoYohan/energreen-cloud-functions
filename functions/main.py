import functions_framework
import google.cloud.firestore
# REMOVE ALL LINES THAT IMPORT Timestamp:
# from google.cloud.firestore import Timestamp
# from google.cloud.firestore_v1.types import Timestamp
# from google.protobuf.timestamp_pb2 import Timestamp as ProtoTimestamp
import datetime # Keep this import
import json
# REMOVE THIS LINE if it's still there: import os

@functions_framework.http
def receive_energy_data(request):
    """
    HTTP Cloud Function to receive energy data from IoT devices and store it in Firestore.
    Expects a POST request with a JSON body containing energy readings.
    """
    # Initialize Firestore client INSIDE the function
    firestore_client = google.cloud.firestore.Client()

    # Ensure it's a POST request
    if request.method != 'POST':
        return ('Method Not Allowed', 405)

    # Parse the JSON request body
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
    timestamp_str = request_json['timestamp']
    kwh_consumed = request_json['kwhConsumed']
    current_amp = request_json['currentAmp']
    voltage_volt = request_json['voltageVolt']
    power_watt = request_json['powerWatt']
    energy_source = request_json.get('energySource', 'Grid') # Default to 'Grid' if not provided

    try:
        # Convert timestamp string to datetime object
        timestamp_dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        # THIS IS THE CRUCIAL CHANGE: Directly use the datetime object
        # Firestore will automatically convert this to its native Timestamp type
        firestore_timestamp_value = timestamp_dt

        # Reference to the specific device's realtime_readings subcollection
        doc_ref = firestore_client.collection('devices').document(device_id).collection('realtime_readings').document(timestamp_str)

        # Prepare data for Firestore
        data_to_store = {
            'timestamp': firestore_timestamp_value, # Pass the datetime object directly
            'kwhConsumed': float(kwh_consumed),
            'currentAmp': float(current_amp),
            'voltageVolt': float(voltage_volt),
            'powerWatt': float(power_watt),
            'energySource': energy_source
        }

        # Store data in Firestore
        doc_ref.set(data_to_store)

        print(f'Data received and stored for device: {device_id} at timestamp: {timestamp_str}')
        return ('Data received successfully!', 200)

    except ValueError as ve:
        print(f'Data conversion error: {ve}')
        return (f'Invalid data format: {ve}', 400)
    except Exception as e:
        print(f'Error writing document to Firestore: {e}')
        return (f'Error processing request and saving data: {e}', 500)

# Ensure no if __name__ == '__main__': block at the end