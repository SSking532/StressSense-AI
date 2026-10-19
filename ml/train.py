import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ml.model import StressModel


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
PATIENCE = 40

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "stress_features.csv"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "stress_model.pth"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


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
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========================================")
print("       STRESSSENSE AI TRAINING")
print("========================================")

print(f"Device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if "subject_id" not in df.columns:
    raise ValueError(
        "The dataset must contain a subject_id column."
    )

print(f"Samples: {len(df)}")
print(f"Participants: {df['subject_id'].nunique()}")


# ============================================================
# FIND TARGET COLUMN
# ============================================================

label_col = next(
    (
        col for col in [
            "stress_label",
            "stress_level",
            "label",
            "target",
        ]
        if col in df.columns
    ),
    None,
)

if label_col is None:
    raise ValueError(
        "No target column found. Expected stress_label, "
        "stress_level, label, or target."
    )

raw_y = df[label_col]

if pd.api.types.is_numeric_dtype(raw_y):
    y = pd.to_numeric(
        raw_y,
        errors="raise",
    ).astype(int).to_numpy()

else:
    normalized = (
        raw_y.astype(str)
        .str.strip()
        .str.lower()
    )

    valid_labels = {
        "rest": 0,
        "stress": 1,
        "0": 0,
        "1": 1,
    }

    y = normalized.map(valid_labels).to_numpy()

    if pd.isna(y).any():
        raise ValueError(
            f"Unrecognized labels in column {label_col}. "
            "Expected Rest/Stress or 0/1."
        )

    y = y.astype(int)

if set(np.unique(y)) - {0, 1}:
    raise ValueError(
        f"Expected binary labels 0/1; got {np.unique(y)}"
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

X = df[FEATURES].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)

groups = df["subject_id"].astype(str).to_numpy()


# ============================================================
# HELD-OUT TEST SPLIT
# ============================================================

# Keep this test split isolated from model selection.
outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED,
)

train_idx, test_idx = next(
    outer_split.split(
        X,
        y,
        groups,
    )
)

X_dev = X.iloc[train_idx].copy()
y_dev = y[train_idx]
groups_dev = groups[train_idx]

X_test = X.iloc[test_idx].copy()
y_test = y[test_idx]
groups_test = groups[test_idx]

assert not (
    set(groups_dev) & set(groups_test)
), "Participant overlap detected in train/test split."


# ============================================================
# TRAIN / VALIDATION SPLIT BY PARTICIPANT
# ============================================================

inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.22,
    random_state=123,
)

fit_idx, val_idx = next(
    inner_split.split(
        X_dev,
        y_dev,
        groups_dev,
    )
)

X_fit = X_dev.iloc[fit_idx].copy()
y_fit = y_dev[fit_idx]
groups_fit = groups_dev[fit_idx]

X_val = X_dev.iloc[val_idx].copy()
y_val = y_dev[val_idx]
groups_val = groups_dev[val_idx]

assert not (
    set(groups_fit) & set(groups_val)
), "Participant overlap detected in train/validation split."

print("\nSplit summary")
print(f"Training samples   : {len(X_fit)}")
print(f"Validation samples : {len(X_val)}")
print(f"Held-out test      : {len(X_test)}")

print(f"Training subjects  : {len(set(groups_fit))}")
print(f"Validation subjects: {len(set(groups_val))}")
print(f"Test subjects      : {len(set(groups_test))}")


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median",
)

X_fit_imp = imputer.fit_transform(X_fit)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)


# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_fit_scaled = scaler.fit_transform(
    X_fit_imp
).astype(np.float32)

X_val_scaled = scaler.transform(
    X_val_imp
).astype(np.float32)

X_test_scaled = scaler.transform(
    X_test_imp
).astype(np.float32)


# ============================================================
# TORCH DATA
# ============================================================

def tensor_x(values):
    return torch.tensor(
        np.asarray(values, dtype=np.float32),
        dtype=torch.float32,
    )


def tensor_y(values):
    return torch.tensor(
        np.asarray(values, dtype=np.int64),
        dtype=torch.long,
    )


train_dataset = TensorDataset(
    tensor_x(X_fit_scaled),
    tensor_y(y_fit),
)

generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_fit,
    minlength=2,
)

class_weights = len(y_fit) / (
    2.0 * np.maximum(class_counts, 1)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)

print("\nClass counts:", class_counts)
print("Class weights:", class_weights.cpu().numpy())


# ============================================================
# MODEL
# ============================================================

model = StressModel(
    input_size=len(FEATURES),
    num_classes=2,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=9,
)


# ============================================================
# TRAINING
# ============================================================

X_val_tensor = tensor_x(
    X_val_scaled
).to(DEVICE)

best_val_f1 = -1.0
best_state = None
best_epoch = 0
epochs_without_improvement = 0

print("\n========================================")
print("           MODEL TRAINING")
print("========================================")

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # Small noise augmentation on training data only.
        augmented_x = (
            batch_x
            + torch.randn_like(batch_x) * 0.01
        )

        optimizer.zero_grad()

        logits = model(augmented_x)

        loss = criterion(
            logits,
            batch_y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item() * len(batch_x)
        )

    # Validation
    model.eval()

    with torch.no_grad():

        val_logits = model(X_val_tensor)

        val_predictions = torch.argmax(
            val_logits,
            dim=1,
        ).cpu().numpy()

    val_accuracy = accuracy_score(
        y_val,
        val_predictions,
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    scheduler.step(val_f1)

    # Select checkpoint using validation F1.
    if val_f1 > best_val_f1:

        best_val_f1 = float(val_f1)
        best_epoch = epoch

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:

        average_loss = (
            total_loss / len(train_dataset)
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}."
        )

        break


# ============================================================
# RESTORE BEST VALIDATION CHECKPOINT
# ============================================================

if best_state is None:
    raise RuntimeError(
        "Training did not produce a valid model checkpoint."
    )

model.load_state_dict(best_state)
model.eval()


# ============================================================
# FINAL HELD-OUT EVALUATION
# ============================================================

with torch.no_grad():

    test_logits = model(
        tensor_x(X_test_scaled).to(DEVICE)
    )

    probabilities = torch.softmax(
        test_logits,
        dim=1,
    ).cpu().numpy()

predictions = probabilities.argmax(axis=1)

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
)

print("\n========================================")
print("       HELD-OUT PARTICIPANT TEST")
print("========================================")

print(f"Best epoch : {best_epoch}")
print(f"Accuracy   : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Rest", "Stress"],
        zero_division=0,
    )
)


# ============================================================
# SAVE MODEL AND SCALER
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH,
)

joblib.dump(
    scaler,
    SCALER_PATH,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "features": FEATURES,
    "input_size": len(FEATURES),
    "num_classes": 2,

    "class_mapping": {
        "0": "Rest",
        "1": "Stress",
    },

    "target_definition": (
        "Project mapping: self-reported stress score <=3 "
        "is Rest; >3 is Stress."
    ),

    "training_samples": int(len(X_fit)),
    "validation_samples": int(len(X_val)),
    "testing_samples": int(len(X_test)),

    "training_subjects": int(len(set(groups_fit))),
    "validation_subjects": int(len(set(groups_val))),
    "testing_subjects": int(len(set(groups_test))),

    "best_epoch": int(best_epoch),
    "validation_f1": float(best_val_f1),

    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),

    "confusion_matrix": cm.tolist(),

    "imputation_strategy": "median",
    "imputation_values": imputer.statistics_.tolist(),

    "device": str(DEVICE),
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


# ============================================================
# SAVE HELD-OUT PREDICTIONS
# ============================================================

test_output = df.iloc[test_idx].copy()

test_output["actual_label"] = y_test
test_output["predicted_label"] = predictions

test_output["confidence"] = (
    probabilities.max(axis=1) * 100
)

test_output["rest_probability"] = (
    probabilities[:, 0] * 100
)

test_output["stress_probability"] = (
    probabilities[:, 1] * 100
)

test_output.to_csv(
    RESULTS_DIR / "test_predictions.csv",
    index=False,
)


# ============================================================
# SAVE EVALUATION JSON
# ============================================================

evaluation = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),

    "classification_report": classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Rest", "Stress"],
        output_dict=True,
        zero_division=0,
    ),
}

with open(
    RESULTS_DIR / "evaluation_results.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        evaluation,
        file,
        indent=4,
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("          TRAINING COMPLETE")
print("========================================")

print(f"Model saved     : {MODEL_PATH}")
print(f"Scaler saved    : {SCALER_PATH}")
print(f"Metadata saved  : {METADATA_PATH}")

print(
    f"Predictions saved: "
    f"{RESULTS_DIR / 'test_predictions.csv'}"
)

print(
    f"\nFinal held-out accuracy: {accuracy * 100:.2f}%"
)

print(
    "The test participants were not used for model selection. "
    "An accuracy target is not guaranteed."
)