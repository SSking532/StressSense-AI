import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "stress_features.csv"
)


MQTT_HOST = os.getenv(
    "HIVEMQ_HOST"
)

MQTT_PORT = int(
    os.getenv(
        "HIVEMQ_PORT",
        8883,
    )
)

MQTT_USERNAME = os.getenv(
    "HIVEMQ_USERNAME"
)

MQTT_PASSWORD = os.getenv(
    "HIVEMQ_PASSWORD"
)

MQTT_TOPIC = os.getenv(
    "HIVEMQ_TOPIC",
    "wearable/stress/data",
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

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
# LOAD REAL DATA
# ============================================================

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )


df = pd.read_csv(
    DATASET_PATH
)


# ============================================================
# IDENTIFY LABEL COLUMN
# ============================================================

LABEL_COLUMN = None

for candidate in [
    "label",
    "stress_level",
    "target",
    "class",
]:

    if candidate in df.columns:

        LABEL_COLUMN = candidate
        break


if LABEL_COLUMN is None:

    raise ValueError(
        "Could not find stress label column."
    )


# ============================================================
# SPLIT REAL DATA
# ============================================================

labels = (
    df[LABEL_COLUMN]
    .astype(str)
    .str.lower()
)


stress_df = df[
    labels.str.contains(
        "stress"
    )
].copy()


rest_df = df[
    ~labels.str.contains(
        "stress"
    )
].copy()


if stress_df.empty or rest_df.empty:

    raise ValueError(
        "Both Rest and Stress samples are required."
    )


# ============================================================
# MQTT CLIENT
# ============================================================

client = mqtt.Client()

client.username_pw_set(
    MQTT_USERNAME,
    MQTT_PASSWORD,
)

client.tls_set()


# ============================================================
# CONNECT
# ============================================================

print(
    "\n========================================"
)

print(
    "      STRESSSENSE LIVE SIMULATOR"
)

print(
    "========================================"
)

print(
    f"Dataset : {DATASET_PATH}"
)

print(
    f"Rest samples   : {len(rest_df)}"
)

print(
    f"Stress samples : {len(stress_df)}"
)

print(
    f"MQTT Topic     : {MQTT_TOPIC}"
)

print(
    "========================================"
)


client.connect(
    MQTT_HOST,
    MQTT_PORT,
    60,
)

client.loop_start()


# ============================================================
# GENERATE PAYLOAD
# ============================================================

def create_payload(
    row,
    scenario,
):

    payload = {}

    for feature in FEATURES:

        value = row.get(
            feature,
            None,
        )

        if pd.isna(value):

            value = 0.0

        payload[
            feature
        ] = float(value)


    payload[
        "device_id"
    ] = "STRESSSENSE-DEMO-01"

    payload[
        "timestamp"
    ] = datetime.now().isoformat()


    # --------------------------------------------------------
    # OPTIONAL ANOMALY INJECTION
    # --------------------------------------------------------

    if scenario == "ANOMALY":

        # We intentionally push several values
        # beyond the project's configured thresholds.

        payload[
            "hr_mean"
        ] = 125.0

        payload[
            "hr_max"
        ] = 135.0

        payload[
            "eda_mean"
        ] = 25.0

        payload[
            "eda_max"
        ] = 30.0

        payload[
            "temp_mean"
        ] = 42.0

        payload[
            "temp_max"
        ] = 43.0

        payload[
            "ibi_mean"
        ] = 0.20

        payload[
            "rmssd"
        ] = 0.002


    return payload


# ============================================================
# LIVE LOOP
# ============================================================

counter = 0

try:

    while True:

        counter += 1


        # ----------------------------------------------------
        # EVERY 6TH SAMPLE = ANOMALY DEMO
        # ----------------------------------------------------

        if counter % 6 == 0:

            row = stress_df.sample(
                1
            ).iloc[0]

            scenario = "ANOMALY"


        # ----------------------------------------------------
        # ALTERNATE REAL REST / STRESS DATA
        # ----------------------------------------------------

        elif counter % 2 == 0:

            row = stress_df.sample(
                1
            ).iloc[0]

            scenario = "STRESS"


        else:

            row = rest_df.sample(
                1
            ).iloc[0]

            scenario = "REST"


        payload = create_payload(
            row,
            scenario,
        )


        message = json.dumps(
            payload
        )


        client.publish(
            MQTT_TOPIC,
            message,
        )


        print(
            "\n----------------------------------------"
        )

        print(
            "LIVE DATA PUBLISHED"
        )

        print(
            "----------------------------------------"
        )

        print(
            f"Scenario    : {scenario}"
        )

        print(
            f"Heart Rate  : {payload['hr_mean']:.2f} BPM"
        )

        print(
            f"EDA         : {payload['eda_mean']:.3f} µS"
        )

        print(
            f"Temperature : {payload['temp_mean']:.2f} °C"
        )

        print(
            f"IBI         : {payload['ibi_mean']:.3f} s"
        )

        print(
            f"RMSSD       : {payload['rmssd']:.3f} s"
        )

        print(
            "----------------------------------------"
        )


        time.sleep(3)


except KeyboardInterrupt:

    print(
        "\nStopping simulator..."
    )

    client.loop_stop()

    client.disconnect()

    print(
        "Disconnected."
    )