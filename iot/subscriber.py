import json
import os

import paho.mqtt.client as mqtt
from dotenv import load_dotenv


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

MQTT_HOST = os.getenv("HIVEMQ_HOST")
MQTT_PORT = int(os.getenv("HIVEMQ_PORT", 8883))

MQTT_USERNAME = os.getenv("HIVEMQ_USERNAME")
MQTT_PASSWORD = os.getenv("HIVEMQ_PASSWORD")

MQTT_TOPIC = os.getenv(
    "HIVEMQ_TOPIC",
    "wearable/stress/data"
)


# ==========================================
# MQTT CALLBACKS
# ==========================================

def on_connect(client, userdata, flags, rc):

    if rc == 0:

        print("\n========================================")
        print("       MQTT DATA SUBSCRIBER")
        print("========================================")
        print("Connected to HiveMQ Cloud")

        client.subscribe(MQTT_TOPIC)

        print(f"Subscribed Topic: {MQTT_TOPIC}")
        print("Waiting for wearable data...\n")

    else:

        print(f"Connection failed. Code: {rc}")


def on_message(client, userdata, msg):

    try:

        data = json.loads(msg.payload.decode("utf-8"))

        print("\nLIVE WEARABLE DATA")
        print("----------------------------------------")

        print(
            f"Device ID        : "
            f"{data.get('device_id', 'Unknown')}"
        )

        print(
            f"Timestamp        : "
            f"{data.get('timestamp', 'Unknown')}"
        )

        print(
            f"Heart Rate       : "
            f"{data.get('hr_mean', 0):.2f} BPM"
        )

        print(
            f"EDA              : "
            f"{data.get('eda_mean', 0):.3f} µS"
        )

        print(
            f"Temperature      : "
            f"{data.get('temp_mean', 0):.2f} °C"
        )

        print(
            f"IBI              : "
            f"{data.get('ibi_mean', 0):.3f} s"
        )

        print(
            f"RMSSD            : "
            f"{data.get('rmssd', 0):.3f} s"
        )

        print(
            f"Acceleration     : "
            f"{data.get('acc_magnitude_mean', 0):.3f}"
        )

        print("----------------------------------------")

    except json.JSONDecodeError:

        print("Error: Received invalid JSON.")

    except Exception as e:

        print(f"Error processing message: {e}")


# ==========================================
# CREATE MQTT CLIENT
# ==========================================

client = mqtt.Client()

client.username_pw_set(
    MQTT_USERNAME,
    MQTT_PASSWORD
)

client.tls_set()

client.on_connect = on_connect
client.on_message = on_message


# ==========================================
# CONNECT
# ==========================================

print("Connecting to HiveMQ Cloud...")

client.connect(
    MQTT_HOST,
    MQTT_PORT,
    60
)


# ==========================================
# START LISTENING
# ==========================================

try:

    client.loop_forever()

except KeyboardInterrupt:

    print("\nStopping subscriber...")

    client.disconnect()

    print("Disconnected from HiveMQ.")