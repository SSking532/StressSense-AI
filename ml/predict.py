import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ml.model import StressModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "stress_model.pth"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
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
# LOAD MODEL COMPONENTS
# ============================================================

_model = None
_scaler = None
_metadata = None


def load_model_components():

    global _model
    global _scaler
    global _metadata

    if _model is not None:
        return _model, _scaler, _metadata

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\n"
            "Run: python -m ml.train"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}\n\n"
            "Run: python -m ml.train"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_PATH}\n\n"
            "Run: python -m ml.train"
        )

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    _scaler = joblib.load(
        SCALER_PATH
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        _metadata = json.load(file)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    _model = StressModel(
        input_size=len(FEATURE_COLUMNS),
        num_classes=2,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        _model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        _model.load_state_dict(
            checkpoint
        )

    _model.eval()

    return _model, _scaler, _metadata


# ============================================================
# PREPROCESS INPUT
# ============================================================

def preprocess_input(data):
    """
    Convert incoming wearable data into the exact
    20-feature format expected by the trained model.
    """

    # --------------------------------------------------------
    # Convert input into DataFrame
    # --------------------------------------------------------

    if isinstance(data, pd.DataFrame):

        input_df = data.copy()

    elif isinstance(data, dict):

        input_df = pd.DataFrame(
            [data]
        )

    else:

        raise TypeError(
            "Input must be a dictionary or pandas DataFrame."
        )

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in input_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # Keep only model features
    # --------------------------------------------------------

    X = input_df[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Load training medians
    # --------------------------------------------------------

    _, _, metadata = load_model_components()

    training_medians = metadata.get(
        "training_medians",
        {}
    )

    if training_medians:

        medians = pd.Series(
            training_medians,
            dtype=float
        )

        X = X.fillna(
            medians.reindex(
                X.columns
            )
        )

    else:

        # Fallback only if metadata does not contain medians.

        X = X.fillna(
            X.median()
        )

    # --------------------------------------------------------
    # Final missing-value check
    # --------------------------------------------------------

    if X.isnull().any().any():

        remaining = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            "Unable to handle missing values in: "
            + ", ".join(remaining)
        )

    return X


# ============================================================
# PREDICT STRESS
# ============================================================

def predict_stress(data):
    """
    Predict Rest or Stress from one wearable feature record.

    Returns a dictionary containing:
        prediction
        stress_level
        confidence
        rest_probability
        stress_probability
    """

    model, scaler, metadata = (
        load_model_components()
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    X = preprocess_input(
        data
    )

    # --------------------------------------------------------
    # Scale using the saved training scaler
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            X_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    rest_probability = float(
        probabilities[0][0].item()
    )

    stress_probability = float(
        probabilities[0][1].item()
    )

    confidence = max(
        rest_probability,
        stress_probability
    )

    # --------------------------------------------------------
    # Class mapping
    # --------------------------------------------------------

    if predicted_class == 1:

        stress_level = "Stress"

    else:

        stress_level = "Rest"

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "prediction": stress_level,
        "stress_level": stress_level,
        "predicted_class": predicted_class,
        "confidence": round(
            confidence * 100,
            2
        ),
        "rest_probability": round(
            rest_probability * 100,
            2
        ),
        "stress_probability": round(
            stress_probability * 100,
            2
        ),
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       STRESSSENSE AI PREDICTION")
    print("========================================")

    print("\nLoading trained model...")

    load_model_components()

    print("✓ Model loaded successfully.")

    # --------------------------------------------------------
    # Use an actual dataset row only as a prediction demo.
    #
    # NOTE:
    # This is NOT training.
    # It simply verifies that the saved model can
    # accept a correctly formatted wearable feature record.
    # --------------------------------------------------------

    DATA_PATH = (
        BASE_DIR
        / "data"
        / "processed"
        / "stress_features.csv"
    )

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    # Pick one record for demonstration.
    sample = df.iloc[0]

    input_data = {
        feature: sample[feature]
        for feature in FEATURE_COLUMNS
    }

    result = predict_stress(
        input_data
    )

    print("\n========================================")
    print("       PREDICTION RESULT")
    print("========================================")

    print(
        f"Prediction       : {result['prediction']}"
    )

    print(
        f"Confidence       : {result['confidence']:.2f}%"
    )

    print(
        f"Rest Probability : "
        f"{result['rest_probability']:.2f}%"
    )

    print(
        f"Stress Probability: "
        f"{result['stress_probability']:.2f}%"
    )

    print("\nInput sample:")
    print(
        f"Subject          : {sample['subject_id']}"
    )

    print(
        f"Stage            : {sample['stage']}"
    )

    if "stress_level" in sample:

        print(
            f"Actual Label     : "
            f"{sample['stress_level']}"
        )

    print("\n✓ Prediction test completed.")