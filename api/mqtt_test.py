import os
import ssl
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HIVEMQ_HOST")
PORT = int(os.getenv("HIVEMQ_PORT", 8883))
USERNAME = os.getenv("HIVEMQ_USERNAME")
PASSWORD = os.getenv("HIVEMQ_PASSWORD")
TOPIC = os.getenv("HIVEMQ_TOPIC")


def on_connect(client, userdata, flags, reason_code, properties):
    print("\n========================================")
    print("MQTT CONNECT CALLBACK")
    print("========================================")
    print("Reason:", reason_code)

    if reason_code == 0:
        print("CONNECTED TO HIVEMQ!")
        client.subscribe(TOPIC)
        print("SUBSCRIBED TO:", TOPIC)


def on_message(client, userdata, msg):
    print("\nMESSAGE RECEIVED:")
    print(msg.payload.decode())


print("========================================")
print("       MQTT CONNECTION TEST")
print("========================================")

print("Host :", HOST)
print("Port :", PORT)
print("Topic:", TOPIC)

client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
)

client.username_pw_set(USERNAME, PASSWORD)

client.tls_set(
    cert_reqs=ssl.CERT_REQUIRED
)

client.on_connect = on_connect
client.on_message = on_message

print("\nConnecting...")

client.connect(HOST, PORT, 60)

print("Starting MQTT loop...")

client.loop_forever()
