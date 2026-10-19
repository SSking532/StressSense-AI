import json
import os

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from ml.predict import predict_stress
from ml.anomaly_detector import detect_anomaly
from database.models import save_prediction


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
# REQUIRED MODEL FEATURES
# ==========================================

REQUIRED_FEATURES = [

    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",

    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",

    "temp_mean",
    "temp_std",
    "temp_min",
    "temp_max",

    "ibi_mean",
    "ibi_std",
    "rmssd",

    "acc_x_mean",
    "acc_y_mean",
    "acc_z_mean",
    "acc_magnitude_mean",
    "acc_magnitude_std",
]


# ==========================================
# MQTT CALLBACK
# ==========================================

def on_connect(client, userdata, flags, rc):

    if rc == 0:

        print("\n========================================")
        print("      STRESSSENSE ML SUBSCRIBER")
        print("========================================")
        print("Connected to HiveMQ Cloud")

        client.subscribe(MQTT_TOPIC)

        print(f"Subscribed Topic: {MQTT_TOPIC}")
        print("Waiting for live wearable data...\n")

    else:

        print(f"MQTT connection failed. Code: {rc}")


# ==========================================
# MESSAGE PROCESSING
# ==========================================

def on_message(client, userdata, msg):

    try:

        # ----------------------------------
        # Decode MQTT message
        # ----------------------------------

        data = json.loads(
            msg.payload.decode("utf-8")
        )


        # ----------------------------------
        # Validate required features
        # ----------------------------------

        missing_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in data
        ]

        if missing_features:

            print("\nERROR")
            print("----------------------------------------")
            print("Missing features:")

            for feature in missing_features:
                print(f"  - {feature}")

            return


        # ----------------------------------
        # Display incoming data
        # ----------------------------------

        print("\n========================================")
        print("        LIVE WEARABLE PREDICTION")
        print("========================================")

        print(
            f"Device ID   : "
            f"{data.get('device_id', 'Unknown')}"
        )

        print(
            f"Timestamp   : "
            f"{data.get('timestamp', 'Unknown')}"
        )

        print("----------------------------------------")

        print(
            f"Heart Rate  : "
            f"{data['hr_mean']:.2f} BPM"
        )

        print(
            f"EDA         : "
            f"{data['eda_mean']:.3f} µS"
        )

        print(
            f"Temperature : "
            f"{data['temp_mean']:.2f} °C"
        )

        print(
            f"IBI         : "
            f"{data['ibi_mean']:.3f} s"
        )

        print(
            f"RMSSD       : "
            f"{data['rmssd']:.3f} s"
        )


        # ==================================
        # STRESS PREDICTION
        # ==================================

        stress_result = predict_stress(data)


        # ==================================
        # ANOMALY DETECTION
        # ==================================

        anomaly_result = detect_anomaly(data)


        # ==================================
        # COMBINE RESULTS
        # ==================================

        result = {
            **stress_result,
            **anomaly_result
        }


        # ==================================
        # SAVE TO DATABASE
        # ==================================

        save_prediction(

            device_id=data.get("device_id"),

            timestamp=data.get("timestamp"),

            prediction=result.get("prediction"),

            stress_level=result.get("stress_level"),

            predicted_class=result.get("predicted_class"),

            confidence=result.get("confidence"),

            rest_probability=result.get("rest_probability"),

            stress_probability=result.get("stress_probability"),

            anomaly_status=result.get("anomaly_status"),

            risk_level=result.get("risk_level"),

            triggered_alerts=result.get(
                "triggered_alerts"
            ),
        )


        # ==================================
        # DISPLAY RESULT
        # ==================================

        print("----------------------------------------")

        print(
            f"Prediction  : "
            f"{result['prediction']}"
        )

        print(
            f"Confidence  : "
            f"{result['confidence']:.2f}%"
        )

        print(
            f"Rest        : "
            f"{result['rest_probability']:.2f}%"
        )

        print(
            f"Stress      : "
            f"{result['stress_probability']:.2f}%"
        )

        print("----------------------------------------")

        print(
            f"Anomaly     : "
            f"{result['anomaly_status']}"
        )

        print(
            f"Risk Level  : "
            f"{result['risk_level']}"
        )


        # ==================================
        # ALERTS
        # ==================================

        alerts = result.get(
            "triggered_alerts",
            []
        )

        if alerts:

            print("\nALERTS:")

            for alert in alerts:

                print(
                    f"  ⚠ {alert}"
                )


        print("========================================")


    except json.JSONDecodeError:

        print(
            "\nERROR: Invalid JSON received."
        )


    except Exception as e:

        print(
            f"\nError processing message: {e}"
        )


# ==========================================
# MQTT CLIENT
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
# START MQTT LOOP
# ==========================================

try:

    client.loop_forever()

except KeyboardInterrupt:

    print("\nStopping ML subscriber...")

    client.disconnect()

    print("Disconnected from HiveMQ.")