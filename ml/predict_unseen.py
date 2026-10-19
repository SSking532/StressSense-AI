import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from ml.model import StressModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "stress_features.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "stress_model.pth"
)

SCALER_PATH = (
    BASE_DIR
    / "models"
    / "scaler.pkl"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "metadata.json"
)

EVALUATION_PATH = (
    BASE_DIR
    / "results"
    / "test_predictions.csv"
)

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    exist_ok=True
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "unseen_predictions.csv"
)


# ============================================================
# FEATURES
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
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "Rest",
    1: "Stress",
}


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("     STRESSSENSE AI UNSEEN PREDICTION")
print("========================================")


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


print("\nLoading trained model...")


scaler = joblib.load(
    SCALER_PATH
)


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(file)


model = StressModel(
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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print("✓ Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

if not EVALUATION_PATH.exists():

    raise FileNotFoundError(
        f"Evaluation predictions not found:\n"
        f"{EVALUATION_PATH}\n\n"
        "Run: python -m ml.evaluate"
    )


df = pd.read_csv(
    EVALUATION_PATH
)


# ============================================================
# VERIFY UNSEEN SUBJECTS
# ============================================================

unseen_subjects = sorted(
    df["subject_id"].unique()
)


print("\n========================================")
print("       UNSEEN SUBJECTS")
print("========================================")

print(
    f"Subjects: {len(unseen_subjects)}"
)

print(
    unseen_subjects
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[
    FEATURE_COLUMNS
].copy()


# Convert to numeric

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

training_medians = metadata.get(
    "training_medians",
    {}
)


if training_medians:

    medians = pd.Series(
        training_medians,
        dtype=float,
    )

    X = X.fillna(
        medians.reindex(
            X.columns
        )
    )


else:

    X = X.fillna(
        X.median()
    )


# ============================================================
# SCALE DATA
# ============================================================

X_scaled = scaler.transform(
    X
)


X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
)


# ============================================================
# PREDICTION
# ============================================================

print("\n========================================")
print("       RUNNING UNSEEN PREDICTIONS")
print("========================================")


with torch.no_grad():

    outputs = model(
        X_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )


predicted_classes = (
    predictions.numpy()
)

probabilities = (
    probabilities.numpy()
)


# ============================================================
# ADD RESULTS
# ============================================================

df["predicted_class"] = (
    predicted_classes
)

df["predicted_label"] = [
    CLASS_NAMES[int(value)]
    for value in predicted_classes
]


df["rest_probability"] = (
    probabilities[:, 0] * 100
)


df["stress_probability"] = (
    probabilities[:, 1] * 100
)


df["prediction_confidence"] = (
    probabilities.max(axis=1) * 100
)


df["prediction_correct"] = (
    df["stress_class"]
    == df["predicted_class"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# OVERALL RESULTS
# ============================================================

correct = int(
    df["prediction_correct"].sum()
)

total = len(df)

accuracy = (
    correct / total
    if total > 0
    else 0
)


print("\n========================================")
print("       UNSEEN DATA RESULTS")
print("========================================")

print(
    f"Unseen samples : {total}"
)

print(
    f"Correct        : {correct}"
)

print(
    f"Incorrect      : {total - correct}"
)

print(
    f"Accuracy       : {accuracy * 100:.2f}%"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n========================================")
print("       SAMPLE PREDICTIONS")
print("========================================")


display_columns = [
    "subject_id",
    "stage",
    "stress_level",
    "predicted_label",
    "prediction_confidence",
    "rest_probability",
    "stress_probability",
]


sample_results = df[
    display_columns
].head(15)


for _, row in sample_results.iterrows():

    print(
        f"\nSubject       : "
        f"{row['subject_id']}"
    )

    print(
        f"Stage         : "
        f"{row['stage']}"
    )

    print(
        f"Actual        : "
        f"{row['stress_level']}"
    )

    print(
        f"Predicted     : "
        f"{row['predicted_label']}"
    )

    print(
        f"Confidence    : "
        f"{row['prediction_confidence']:.2f}%"
    )

    print(
        f"Rest Prob.    : "
        f"{row['rest_probability']:.2f}%"
    )

    print(
        f"Stress Prob.  : "
        f"{row['stress_probability']:.2f}%"
    )


# ============================================================
# SUBJECT-WISE RESULTS
# ============================================================

print("\n========================================")
print("       SUBJECT-WISE PREDICTION")
print("========================================")


subject_results = (
    df
    .groupby("subject_id")
    .agg(
        samples=("subject_id", "count"),
        correct=("prediction_correct", "sum"),
        average_confidence=(
            "prediction_confidence",
            "mean",
        ),
    )
    .reset_index()
)


subject_results["accuracy"] = (
    subject_results["correct"]
    / subject_results["samples"]
    * 100
)


for _, row in subject_results.iterrows():

    print(
        f"{row['subject_id']:>4} | "
        f"Samples: {int(row['samples']):2d} | "
        f"Accuracy: {row['accuracy']:6.2f}% | "
        f"Confidence: "
        f"{row['average_confidence']:6.2f}%"
    )


# ============================================================
# FINAL
# ============================================================

print("\n========================================")
print("       PREDICTION COMPLETE")
print("========================================")

print(
    f"Results saved to:\n{OUTPUT_PATH}"
)

print(
    "\n✓ Predictions were generated for "
    "subjects that were not used during "
    "model training."
)