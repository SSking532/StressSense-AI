import json
import math
import os

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from ml.predict import predict_stress
from ml.anomaly_detector import detect_anomaly
from database.models import save_prediction


# ============================================================
# MQTT CONFIGURATION
# ============================================================

load_dotenv()

MQTT_HOST = os.getenv("HIVEMQ_HOST")

MQTT_PORT = int(
    os.getenv("HIVEMQ_PORT", "8883")
)

MQTT_USERNAME = os.getenv("HIVEMQ_USERNAME")
MQTT_PASSWORD = os.getenv("HIVEMQ_PASSWORD")

MQTT_TOPIC = os.getenv(
    "HIVEMQ_TOPIC",
    "wearable/stress/data",
)


# ============================================================
# REQUIRED MODEL FEATURES
# ============================================================

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


# ============================================================
# MQTT CLIENT
# ============================================================

class MQTTClient:

    def __init__(self):

        if not MQTT_HOST:
            raise ValueError(
                "HIVEMQ_HOST is missing from the .env file."
            )

        self.client = mqtt.Client()

        if MQTT_USERNAME and MQTT_PASSWORD:
            self.client.username_pw_set(
                MQTT_USERNAME,
                MQTT_PASSWORD,
            )

        self.client.tls_set()

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    # ========================================================
    # CONNECT
    # ========================================================

    def on_connect(
        self,
        client,
        userdata,
        flags,
        rc,
        *extra,
    ):

        reason_code = getattr(rc, "value", rc)

        if reason_code == 0:

            print("\n========================================")
            print("          MQTT CONNECTED")
            print("========================================")

            client.subscribe(MQTT_TOPIC)

            print(f"Subscribed: {MQTT_TOPIC}")

        else:

            print(
                f"MQTT connection failed. Code: {rc}"
            )

    # ========================================================
    # RECEIVE AND PROCESS DATA
    # ========================================================

    def on_message(
        self,
        client,
        userdata,
        msg,
    ):

        print("\n----------------------------------------")
        print("       WEARABLE DATA RECEIVED")
        print("----------------------------------------")

        try:

            # Decode JSON
            data = json.loads(
                msg.payload.decode("utf-8")
            )

            # ------------------------------------------------
            # VALIDATE FEATURES
            # ------------------------------------------------

            missing = [
                feature
                for feature in REQUIRED_FEATURES
                if feature not in data
            ]

            if missing:

                print(
                    "Message skipped. Missing features:"
                )

                for feature in missing:
                    print(f"  - {feature}")

                return

            for feature in REQUIRED_FEATURES:

                value = data[feature]

                if (
                    value is None
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):

                    print(
                        f"Message skipped. Invalid value: {feature}"
                    )

                    return

            # ------------------------------------------------
            # DISPLAY SENSOR DATA
            # ------------------------------------------------

            print(
                f"Device ID   : {data.get('device_id', 'Unknown')}"
            )

            print(
                f"Timestamp   : {data.get('timestamp', 'Unavailable')}"
            )

            print(
                f"Heart Rate  : {float(data['hr_mean']):.2f} BPM"
            )

            print(
                f"EDA         : {float(data['eda_mean']):.3f} µS"
            )

            print(
                f"Temperature : {float(data['temp_mean']):.2f} °C"
            )

            print(
                f"IBI         : {float(data['ibi_mean']):.3f} seconds"
            )

            print(
                f"RMSSD       : {float(data['rmssd']):.3f} seconds"
            )

            # ------------------------------------------------
            # RUN STRESS MODEL
            # ------------------------------------------------

            stress_result = predict_stress(data)

            # ------------------------------------------------
            # RUN ANOMALY DETECTOR
            # ------------------------------------------------

            anomaly_result = detect_anomaly(data)

            # ------------------------------------------------
            # COMBINE RESULTS
            # ------------------------------------------------

            result = {
                **stress_result,
                **anomaly_result,
            }

            # ------------------------------------------------
            # SAVE TO DATABASE
            # ------------------------------------------------

            save_prediction(
                device_id=data.get("device_id"),

                timestamp=data.get("timestamp"),

                prediction=result.get(
                    "prediction",
                    "Unknown",
                ),

                stress_level=result.get(
                    "stress_level",
                    result.get("prediction", "Unknown"),
                ),

                predicted_class=int(
                    result.get("predicted_class", 0)
                ),

                confidence=float(
                    result.get("confidence", 0)
                ),

                rest_probability=float(
                    result.get("rest_probability", 0)
                ),

                stress_probability=float(
                    result.get("stress_probability", 0)
                ),

                anomaly_status=result.get(
                    "anomaly_status",
                    "NORMAL",
                ),

                risk_level=result.get(
                    "risk_level",
                    "LOW",
                ),

                triggered_alerts=(
                    result.get("triggered_alerts") or []
                ),
            )

            # ------------------------------------------------
            # DISPLAY MODEL RESULT
            # ------------------------------------------------

            print("\n========================================")
            print("          PREDICTION RESULT")
            print("========================================")

            print(
                f"Prediction  : {result.get('prediction')}"
            )

            print(
                f"Confidence  : "
                f"{float(result.get('confidence', 0)):.2f}%"
            )

            print(
                f"Rest        : "
                f"{float(result.get('rest_probability', 0)):.2f}%"
            )

            print(
                f"Stress      : "
                f"{float(result.get('stress_probability', 0)):.2f}%"
            )

            print(
                f"Anomaly     : {result.get('anomaly_status')}"
            )

            print(
                f"Risk Level  : {result.get('risk_level')}"
            )

            # ------------------------------------------------
            # PRINT ALERTS
            # ------------------------------------------------

            alerts = result.get("triggered_alerts") or []

            if alerts:

                print("\n🚨 ANOMALY ALERTS")

                for alert in alerts:
                    print(f"  ⚠️ {alert}")

            print("========================================")

        except json.JSONDecodeError:

            print(
                "MQTT ERROR: Invalid JSON received."
            )

        except Exception as exc:

            print(
                f"MQTT PROCESSING ERROR: {exc}"
            )

    # ========================================================
    # START
    # ========================================================

    def start(self):

        print("\n========================================")
        print("         STARTING MQTT SERVICE")
        print("========================================")

        print(f"Host  : {MQTT_HOST}")
        print(f"Port  : {MQTT_PORT}")
        print(f"Topic : {MQTT_TOPIC}")

        self.client.connect(
            MQTT_HOST,
            MQTT_PORT,
            60,
        )

        self.client.loop_start()

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        print("\nStopping MQTT service...")

        self.client.loop_stop()
        self.client.disconnect()

        print("MQTT disconnected.")