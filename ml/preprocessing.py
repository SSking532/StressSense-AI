import os
import joblib
import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)


FEATURES = [
    "heart_rate",
    "hrv",
    "respiration_rate",
    "activity_level",
    "sleep_hours"
]


# ==========================================
# LOAD TRAINED SCALER
# ==========================================

def load_scaler():

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    return joblib.load(SCALER_PATH)


# ==========================================
# VALIDATE INPUT
# ==========================================

def validate_input(data):

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )


# ==========================================
# PREPROCESS LIVE DATA
# ==========================================

def preprocess_input(data):

    validate_input(data)

    values = [
        data["heart_rate"],
        data["hrv"],
        data["respiration_rate"],
        data["activity_level"],
        data["sleep_hours"]
    ]

    values = np.array(
        values,
        dtype=np.float32
    ).reshape(1, -1)

    scaler = load_scaler()

    scaled_values = scaler.transform(values)

    return scaled_values.astype(np.float32)