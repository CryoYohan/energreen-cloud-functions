import paho.mqtt.client as mqtt
from google.cloud import pubsub_v1
import os
import time
import threading
import http.server
import socketserver
import sys

print("Bridge script started", flush=True)

# Configuration (set via environment variables)
MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = os.getenv("MQTT_PORT")
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
PUBSUB_PROJECT = os.getenv("PUBSUB_PROJECT")
PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC")
PORT = int(os.getenv("PORT", 8080))  # Default to 8080 for Cloud Run

# Validate environment variables
required_vars = {
    "MQTT_BROKER": MQTT_BROKER,
    "MQTT_PORT": MQTT_PORT,
    "MQTT_USER": MQTT_USER,
    "MQTT_PASS": MQTT_PASS,
    "MQTT_TOPIC": MQTT_TOPIC,
    "PUBSUB_PROJECT": PUBSUB_PROJECT,
    "PUBSUB_TOPIC": PUBSUB_TOPIC
}
for var_name, var_value in required_vars.items():
    if not var_value:
        print(f"Error: Environment variable {var_name} is not set", flush=True)
        raise ValueError(f"Missing environment variable: {var_name}")

MQTT_PORT = int(MQTT_PORT)

# Initialize Pub/Sub publisher
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PUBSUB_PROJECT, PUBSUB_TOPIC)

# MQTT callbacks
def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"MQTT connect result: reason_code={reason_code}", flush=True)
    if reason_code == 0:
        print("Connected to MQTT broker", flush=True)
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to {MQTT_TOPIC}", flush=True)
    else:
        print(f"Failed to connect: reason_code={reason_code}", flush=True)

def on_message(client, userdata, msg):
    try:
        print(f"Received: topic={msg.topic}, payload={msg.payload.decode(errors='ignore')[:100]}", flush=True)
        payload = msg.payload
        publisher.publish(topic_path, data=payload)
        print(f"Forwarded to Pub/Sub: {msg.topic} - {len(payload)} bytes", flush=True)
    except Exception as e:
        print(f"Error publishing to Pub/Sub: {e}", flush=True)

# Simple HTTP server for Cloud Run health checks
class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

def run_mqtt_client():
    print("Starting MQTT client thread", flush=True)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    ca_path = os.path.abspath("isrgrootx1.pem")
    print(f"Checking CA cert at: {ca_path}", flush=True)
    if not os.path.exists(ca_path):
        print(f"Error: CA cert not found at {ca_path}. Falling back to system defaults.", flush=True)
        client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
    else:
        print(f"CA cert found, size: {os.path.getsize(ca_path)} bytes", flush=True)
        try:
            client.tls_set(ca_certs="isrgrootx1.pem", tls_version=mqtt.ssl.PROTOCOL_TLS)
            print("TLS set successfully with custom CA", flush=True)
        except Exception as e:
            print(f"TLS setup failed: {e}. Falling back to system defaults.", flush=True)
            client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
    print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT} with topic {MQTT_TOPIC}", flush=True)
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print("Connection attempt made", flush=True)
    except Exception as e:
        print(f"Connection failed: {e}", flush=True)
    client.loop_forever()

def run_http_server():
    httpd = socketserver.TCPServer(("", PORT), HealthCheckHandler)
    print(f"Starting HTTP health check server on port {PORT}", flush=True)
    httpd.serve_forever()

def main():
    print("Entering main: Starting threads", flush=True)
    # Run MQTT client in a separate thread
    mqtt_thread = threading.Thread(target=run_mqtt_client)
    mqtt_thread.daemon = True
    mqtt_thread.start()

    # Run HTTP server in the main thread (Cloud Run expects this)
    run_http_server()

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"Bridge error: {e}", flush=True)
            time.sleep(5)  # Retry after delay