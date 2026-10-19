import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit

from ml.model import StressModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "stress_features.csv"
MODEL_PATH = BASE_DIR / "models" / "stress_model.pth"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "evaluation_results.json"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"


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
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

CLASS_NAMES = {
    0: "Rest",
    1: "Stress",
}


# ============================================================
# HEADER
# ============================================================

print("\n========================================")
print("       STRESSSENSE AI EVALUATION")
print("========================================")


# ============================================================
# CHECK FILES
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{DATA_PATH}\n\n"
        "Run: python -m ml.dataset_loader"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}\n\n"
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


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded : {len(df)} samples")
print(f"Subjects       : {df['subject_id'].nunique()}")
print(f"Features       : {len(FEATURE_COLUMNS)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURE_COLUMNS + [
    "subject_id",
    "stress_class",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns:\n{missing_columns}"
    )


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURE_COLUMNS].copy()

y = df["stress_class"].copy()

groups = df["subject_id"].copy()


# ============================================================
# SUBJECT-WISE SPLIT
# ============================================================

print("\n========================================")
print("       SUBJECT-WISE DATA SPLIT")
print("========================================")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups=groups,
    )
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()

train_subjects = sorted(
    df.iloc[train_indices]["subject_id"].unique()
)

test_subjects = sorted(
    df.iloc[test_indices]["subject_id"].unique()
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

print(f"Training subjects: {len(train_subjects)}")
print(f"Testing subjects : {len(test_subjects)}")

print("\nTraining subjects:")
print(train_subjects)

print("\nTesting subjects:")
print(test_subjects)


# ============================================================
# CHECK LEAKAGE
# ============================================================

overlap = set(train_subjects).intersection(
    set(test_subjects)
)

if overlap:
    raise RuntimeError(
        f"Subject leakage detected: {sorted(overlap)}"
    )

print("\n✓ No subject overlap between training and testing sets.")


# ============================================================
# LOAD SAVED SCALER
# ============================================================

print("\n========================================")
print("       LOADING PREPROCESSING")
print("========================================")

# IMPORTANT:
# train.py uses joblib.dump(), so the scaler
# must be loaded using joblib.load().

scaler = joblib.load(SCALER_PATH)

print("✓ Scaler loaded.")


# ============================================================
# LOAD TRAINING MEDIANS
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

training_medians = metadata.get(
    "training_medians",
    {}
)


# ============================================================
# IMPUTE MISSING VALUES
# ============================================================

if training_medians:

    medians = pd.Series(
        training_medians,
        dtype=float,
    )

    X_train = X_train.fillna(
        medians.reindex(X_train.columns)
    )

    X_test = X_test.fillna(
        medians.reindex(X_test.columns)
    )

    print(
        "✓ Training-set median imputation "
        "loaded from metadata."
    )

else:

    print(
        "⚠ Training medians not found in metadata."
    )

    print(
        "Using training-set median imputation."
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=FEATURE_COLUMNS,
        index=X_train.index,
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=FEATURE_COLUMNS,
        index=X_test.index,
    )


# ============================================================
# SCALE TEST FEATURES
# ============================================================

X_test_scaled = scaler.transform(X_test)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
)

y_test_tensor = torch.tensor(
    y_test.to_numpy(),
    dtype=torch.long,
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\n========================================")
print("       LOADING TRAINED MODEL")
print("========================================")

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

print("✓ PyTorch model loaded.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n========================================")
print("       GENERATING PREDICTIONS")
print("========================================")

with torch.no_grad():

    outputs = model(
        X_test_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    predictions = torch.argmax(
        outputs,
        dim=1,
    )

y_true = y_test_tensor.numpy()

y_pred = predictions.numpy()

probabilities_np = probabilities.numpy()


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0,
)

conf_matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

classification_report_dict = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=[
        "Rest",
        "Stress",
    ],
    output_dict=True,
    zero_division=0,
)


# ============================================================
# DISPLAY PERFORMANCE
# ============================================================

print("\n========================================")
print("       MODEL PERFORMANCE")
print("========================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


print("\nConfusion Matrix:")

print(conf_matrix)


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Rest",
            "Stress",
        ],
        zero_division=0,
    )
)


# ============================================================
# CREATE TEST PREDICTION DATAFRAME
# ============================================================

results_df = df.iloc[
    test_indices
].copy()

results_df["actual_class"] = y_true

results_df["predicted_class"] = y_pred

results_df["actual_label"] = [
    CLASS_NAMES[int(value)]
    for value in y_true
]

results_df["predicted_label"] = [
    CLASS_NAMES[int(value)]
    for value in y_pred
]

results_df["rest_probability"] = (
    probabilities_np[:, 0]
)

results_df["stress_probability"] = (
    probabilities_np[:, 1]
)

results_df["correct_prediction"] = (
    results_df["actual_class"]
    == results_df["predicted_class"]
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

results_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

print("\n✓ Test predictions saved:")

print(PREDICTIONS_PATH)


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

evaluation_results = {

    "dataset": {

        "total_samples": int(
            len(df)
        ),

        "total_subjects": int(
            df["subject_id"].nunique()
        ),

        "feature_count": len(
            FEATURE_COLUMNS
        ),
    },

    "split": {

        "method": "GroupShuffleSplit",

        "test_size": TEST_SIZE,

        "random_state": RANDOM_STATE,

        "training_samples": int(
            len(train_indices)
        ),

        "testing_samples": int(
            len(test_indices)
        ),

        "training_subjects": train_subjects,

        "testing_subjects": test_subjects,

        "subject_overlap": sorted(
            list(overlap)
        ),
    },

    "target": {

        "0": "Rest",

        "1": "Stress",

        "definition": (
            "Binary target derived from the "
            "dataset's self-reported stress "
            "score: score <= 3 mapped to Rest "
            "and score > 3 mapped to Stress."
        ),
    },

    "metrics": {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1_score": float(
            f1
        ),
    },

    "confusion_matrix": (
        conf_matrix.tolist()
    ),

    "classification_report": (
        classification_report_dict
    ),

    "feature_columns": FEATURE_COLUMNS,

    "model": {

        "architecture": (
            "PyTorch Feed-Forward "
            "Neural Network"
        ),

        "input_size": len(
            FEATURE_COLUMNS
        ),

        "hidden_layers": [
            64,
            32,
            16,
        ],

        "output_classes": 2,
    },
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        evaluation_results,
        file,
        indent=4,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("       EVALUATION COMPLETE")
print("========================================")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print("\nSaved results:")

print(
    f"JSON : {RESULTS_PATH}"
)

print(
    f"CSV  : {PREDICTIONS_PATH}"
)

print(
    "\n✓ Evaluation finished successfully."
)