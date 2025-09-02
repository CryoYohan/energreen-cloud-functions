import functions_framework
import google.cloud.firestore
import datetime
import json
import math

# -------------------------------
# Normalization Helper
# -------------------------------
def normalize_reading(reading):
    """Normalize fields to expected units: V, A, W, kWh, Hz, PF."""
    r = reading.copy()

    # Voltage sanity
    if r.get('voltageVolt', 0) > 1000:
        r['voltageVolt'] = r['voltageVolt'] / 10.0

    # Current sanity
    if r.get('currentAmp', 0) > 1000:
        r['currentAmp'] = r['currentAmp'] / 1000.0

    # Power sanity
    if r.get('powerWatt', 0) > 100000:
        r['powerWatt'] = r['powerWatt'] / 1000.0

    # Energy sanity (Wh vs kWh confusion)
    if r.get('kwhConsumed', 0) > 10000:
        r['kwhConsumed'] = r['kwhConsumed'] / 1000.0

    return r

# -------------------------------
# Improved Appliance Prediction
# -------------------------------
def predict_appliance_type(signature_data):
    """
    A simple rule-based model to predict appliance type based on
    average power consumption AND duration of signature.
    """
    if not signature_data:
        return 'Unknown'

    # Normalize before prediction
    normalized = [normalize_reading(d) for d in signature_data]

    total_power = sum(float(d.get('powerWatt', 0)) for d in normalized)
    average_power = total_power / len(normalized)

    # Estimate event duration
    timestamps = [d.get("timestamp", 0) for d in normalized if "timestamp" in d]
    duration = (max(timestamps) - min(timestamps)) if timestamps else 0

    # Prediction rules: both power & duration
    if average_power > 1500:
        return 'Oven'
    elif average_power > 1000:
        return 'Electric Kettle'
    elif average_power > 500:
        return 'Microwave'
    elif average_power > 100:
        if duration < 30:
            return 'Toaster'
        else:
            return 'Fan / Small Appliance'
    elif average_power > 50:
        return 'Lightbulb'
    else:
        return 'Standby Power'

# -------------------------------
# Cloud Function Entry Point
# -------------------------------
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
        EPOCH_OFFSET_SECONDS_1970_TO_2000 = 946684800

        # -------------------------------
        # Appliance Signatures
        # -------------------------------
        if data_type == 'ApplianceSignature':
            if 'signature_data' not in request_json:
                return ('Missing required data field: signature_data for ApplianceSignature type.', 400)

            normalized_signature = [normalize_reading(d) for d in request_json['signature_data']]

            # Use the timestamp from the first reading
            first_reading_timestamp_mpy = normalized_signature[0]['timestamp']
            unix_timestamp_seconds = first_reading_timestamp_mpy + EPOCH_OFFSET_SECONDS_1970_TO_2000
            timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)
            doc_id = timestamp_dt.isoformat(timespec='seconds').replace('+00:00', 'Z').replace(':', '-')

            # 1. Store raw signature data
            data_to_store = {
                'deviceId': device_id,
                'timestamp': timestamp_dt,
                'signature_data': normalized_signature,
                'processed_at': datetime.datetime.now(tz=datetime.timezone.utc)
            }
            doc_ref = firestore_client.collection('devices').document(device_id).collection('appliance_signatures').document(doc_id)
            doc_ref.set(data_to_store)
            print(f'Appliance signature stored for device: {device_id} at {timestamp_dt.isoformat()}')

            # 2. Predict appliance type
            predicted_appliance = predict_appliance_type(normalized_signature)
            prediction_data = {
                'deviceId': device_id,
                'timestamp': timestamp_dt,
                'predictedAppliance': predicted_appliance,
                'processed_at': datetime.datetime.now(tz=datetime.timezone.utc)
            }
            prediction_doc_ref = firestore_client.collection('devices').document(device_id).collection('predicted_appliances').document(doc_id)
            prediction_doc_ref.set(prediction_data)
            print(f'Prediction stored for device: {device_id}, Predicted appliance: {predicted_appliance}')

        # -------------------------------
        # Regular Readings
        # -------------------------------
        elif data_type == 'RegularReading':
            required_fields = ['timestamp', 'kwhConsumed', 'currentAmp', 'voltageVolt', 'powerWatt']
            for field in required_fields:
                if field not in request_json:
                    print(f'Missing required field: {field} in {request_json}')
                    return (f'Missing required data field: {field}.', 400)

            normalized = normalize_reading(request_json)

            timestamp_mpy_int = normalized['timestamp']
            unix_timestamp_seconds = timestamp_mpy_int + EPOCH_OFFSET_SECONDS_1970_TO_2000
            timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)
            doc_id = timestamp_dt.isoformat(timespec='seconds').replace('+00:00', 'Z').replace(':', '-')

            data_to_store = {
                'timestamp': timestamp_dt,
                'kwhConsumed': float(normalized['kwhConsumed']),
                'currentAmp': float(normalized['currentAmp']),
                'voltageVolt': float(normalized['voltageVolt']),
                'powerWatt': float(normalized['powerWatt']),
                'energySource': normalized.get('energySource', 'Grid'),
                'powerFactor': normalized.get('powerFactor', None),
                'timestamp_esp32_raw': timestamp_mpy_int
            }
            doc_ref = firestore_client.collection('devices').document(device_id).collection('realtime_readings').document(doc_id)
            doc_ref.set(data_to_store)
            print(f'Regular reading stored for device: {device_id} at {timestamp_dt.isoformat()}')

        else:
            return ('Invalid dataType provided.', 400)

        return ('Data received and processed successfully!', 200)

    except (ValueError, TypeError) as ve:
        print(f'Data conversion or format error: {ve}')
        return (f'Invalid data format: {ve}', 400)
    except Exception as e:
        print(f'Error processing request and saving data: {e}')
        return (f'Error processing request and saving data: {e}', 500)
