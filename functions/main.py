import functions_framework
import google.cloud.firestore
import datetime
import json
import math
import sys

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
# Cloud Function Entry Point
# -------------------------------
@functions_framework.http
def receive_energy_data(request):
    """
    HTTP Cloud Function to receive energy data from IoT devices and store it in Firestore.
    Handles both RegularReading and ApplianceSignature.
    """
    try:
        firestore_client = google.cloud.firestore.Client()
    except Exception as e:
        print(f"Firestore client initialization failed: {e}", file=sys.stderr)
        return ('Internal Server Error: Firestore client failed to initialize.', 500)

    if request.method != 'POST':
        return ('Method Not Allowed', 405)

    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return ('Request body must be JSON', 400)
    except Exception as e:
        return (f'Error parsing JSON: {e}', 400)

    if 'dataType' not in request_json or 'deviceId' not in request_json:
        return ('Missing required data fields: dataType or deviceId.', 400)

    data_type = request_json['dataType']
    device_id = request_json['deviceId']

    EPOCH_OFFSET_SECONDS_1970_TO_2000 = 946684800

    try:
        # -------------------------------
        # Appliance Signatures
        # -------------------------------
        if data_type == 'ApplianceSignature':
            if 'signature_data' not in request_json:
                return ('Missing required data field: signature_data for ApplianceSignature.', 400)

            normalized_signature = [normalize_reading(d) for d in request_json['signature_data']]
            first_reading_timestamp_mpy = normalized_signature[0]['timestamp']
            unix_timestamp_seconds = first_reading_timestamp_mpy + EPOCH_OFFSET_SECONDS_1970_TO_2000
            timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)

            # Step 1: Store raw signature in appliance_predictions
            prediction_ref = firestore_client.collection("devices") \
                .document(device_id) \
                .collection("appliance_predictions") \
                .document()  # auto-ID

            prediction_doc = {
                "deviceId": device_id,
                "timestamp": timestamp_dt,
                "signature": normalized_signature,
                "event_type": request_json.get("event_type", "unknown"),

                # 🔹 Lifecycle fields
                "status": "unidentified",   # starts as unidentified
                "cluster_id": None,         # assigned later by clustering
                "predicted_label": None,    # clustering/ML adds this later
                "confidence": None,
                "user_label": None,         # set only when user confirms
                "confirmed_at": None,
                "created_at": datetime.datetime.now(tz=datetime.timezone.utc)
            }
            prediction_ref.set(prediction_doc)

            print(f'Appliance signature stored: {prediction_ref.id}')


        # -------------------------------
        # Regular Readings
        # -------------------------------
        elif data_type == 'RegularReading':
            required_fields = ['timestamp', 'kwhConsumed', 'currentAmp', 'voltageVolt', 'powerWatt']
            for field in required_fields:
                if field not in request_json:
                    return (f'Missing required data field: {field}.', 400)

            normalized = normalize_reading(request_json)
            timestamp_mpy_int = normalized['timestamp']
            unix_timestamp_seconds = timestamp_mpy_int + EPOCH_OFFSET_SECONDS_1970_TO_2000
            timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)

            doc_id = timestamp_dt.isoformat(timespec='seconds').replace('+00:00', 'Z').replace(':', '-')
            doc_ref = firestore_client.collection('devices').document(device_id).collection('realtime_readings').document(doc_id)

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

            doc_ref.set(data_to_store)
            print(f'Regular reading stored at {doc_id}')

        else:
            return ('Invalid dataType provided.', 400)

    except Exception as e:
        print(f'Error processing data: {e}', file=sys.stderr)
        return (f'Error processing data: {e}', 500)

    return ('Data received and processed successfully!', 200)
