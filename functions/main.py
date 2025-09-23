import functions_framework
import google.cloud.firestore
import datetime
import json
import math
import statistics
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
# Feature Extraction Helpers
# -------------------------------
def summarize_transient(data):
    """Extract transient summary features from normalized list of readings."""
    if not data:
        return None
    powers = [d['powerWatt'] for d in data]
    timestamps = [d['timestamp'] for d in data]

    peak_power = max(powers)
    energy = sum(powers) * 1  # assuming 1s sample spacing (adjust if ESP32 sends faster)
    rise_time = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0

    return {
        "peak_power": float(peak_power),
        "energy": float(energy),
        "rise_time": float(rise_time)
    }


def summarize_steady(data):
    """Extract steady-state summary features from normalized list of readings."""
    if not data:
        return None
    powers = [d['powerWatt'] for d in data]
    currents = [d['currentAmp'] for d in data]
    pfs = [d.get('powerFactor', 0.0) for d in data]
    voltages = [d['voltageVolt'] for d in data]

    return {
        "mean_power": float(statistics.mean(powers)),
        "std_power": float(statistics.pstdev(powers)) if len(powers) > 1 else 0.0,
        "pf_mean": float(statistics.mean(pfs)),
        "current_rms": float(statistics.mean(currents)),
        "voltage_mean": float(statistics.mean(voltages)),
        "voltage_std": float(statistics.pstdev(voltages)) if len(voltages) > 1 else 0.0
    }


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
            if 'transient_data' not in request_json or 'steady_state_data' not in request_json:
                return ('Missing required data fields: transient_data or steady_state_data.', 400)

            normalized_transient = [normalize_reading(d) for d in request_json['transient_data']]
            normalized_steady = [normalize_reading(d) for d in request_json['steady_state_data']]

            # summaries
            transient_summary = summarize_transient(normalized_transient)
            steady_summary = summarize_steady(normalized_steady)

            first_reading_timestamp_mpy = normalized_transient[0]['timestamp']
            unix_timestamp_seconds = first_reading_timestamp_mpy + EPOCH_OFFSET_SECONDS_1970_TO_2000
            timestamp_dt = datetime.datetime.fromtimestamp(unix_timestamp_seconds, tz=datetime.timezone.utc)

            prediction_ref = firestore_client.collection("devices") \
                .document(device_id) \
                .collection("appliance_predictions") \
                .document()  # auto-ID

            prediction_doc = {
                "deviceId": device_id,
                "timestamp": timestamp_dt,
                "event_type": request_json.get("event_type", "unknown"),
                "transient_data": normalized_transient,
                "steady_state_data": normalized_steady,
                "transient_summary": transient_summary,
                "steady_summary": steady_summary,
                # lifecycle fields
                "status": "unidentified",
                "cluster_id": None,
                "predicted_label": None,
                "confidence": None,
                "user_label": "LED Bulb 3W",
                "confirmed_at": None,
                "created_at": datetime.datetime.now(tz=datetime.timezone.utc),
                "capture_settings": {
                    "sample_interval_s": 1,
                    "event_duration_s": len(normalized_transient) + len(normalized_steady)
                }
            }

            prediction_ref.set(prediction_doc)
            print(f'Appliance signature stored with summaries: {prediction_ref.id}')


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
