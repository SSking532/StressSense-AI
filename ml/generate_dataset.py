import os
import numpy as np
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

NUM_SAMPLES = 10000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ==========================================
# CREATE DATA DIRECTORY
# ==========================================

DATA_DIR = "data/raw"

os.makedirs(DATA_DIR, exist_ok=True)


# ==========================================
# GENERATE BASIC FEATURES
# ==========================================

# Heart Rate (BPM)
heart_rate = np.random.normal(
    loc=78,
    scale=12,
    size=NUM_SAMPLES
)

heart_rate = np.clip(
    heart_rate,
    55,
    130
)


# Heart Rate Variability (ms)
hrv = np.random.normal(
    loc=55,
    scale=15,
    size=NUM_SAMPLES
)

hrv = np.clip(
    hrv,
    15,
    100
)


# Respiration Rate (breaths/min)
respiration_rate = np.random.normal(
    loc=16,
    scale=3,
    size=NUM_SAMPLES
)

respiration_rate = np.clip(
    respiration_rate,
    10,
    30
)


# Activity Level (0-1)
activity_level = np.random.uniform(
    0.05,
    1.0,
    NUM_SAMPLES
)


# Sleep Duration (hours)
sleep_hours = np.random.normal(
    loc=7,
    scale=1.2,
    size=NUM_SAMPLES
)

sleep_hours = np.clip(
    sleep_hours,
    3.5,
    10
)


# ==========================================
# CALCULATE STRESS SCORE
# ==========================================

stress_score = (
    0.050 * heart_rate
    - 0.040 * hrv
    + 0.120 * respiration_rate
    - 0.350 * sleep_hours
    + 0.150 * activity_level
)


# ==========================================
# ADD NATURAL VARIATION
# ==========================================

noise = np.random.normal(
    0,
    0.15,
    NUM_SAMPLES
)

stress_score += noise


# ==========================================
# NORMALIZE STRESS SCORE
# ==========================================

stress_score = (
    stress_score - stress_score.min()
) / (
    stress_score.max() - stress_score.min()
)


# ==========================================
# CREATE STRESS LEVELS
# ==========================================

# 0 = Low
# 1 = Moderate
# 2 = High

stress_level = np.where(
    stress_score < 0.33,
    0,
    np.where(
        stress_score < 0.66,
        1,
        2
    )
)


# ==========================================
# CREATE DATAFRAME
# ==========================================

data = pd.DataFrame({

    "heart_rate": np.round(
        heart_rate,
        2
    ),

    "hrv": np.round(
        hrv,
        2
    ),

    "respiration_rate": np.round(
        respiration_rate,
        2
    ),

    "activity_level": np.round(
        activity_level,
        2
    ),

    "sleep_hours": np.round(
        sleep_hours,
        2
    ),

    "stress_score": np.round(
        stress_score,
        4
    ),

    "stress_level": stress_level
})


# ==========================================
# SAVE DATASET
# ==========================================

output_path = os.path.join(
    DATA_DIR,
    "synthetic_wearable_data.csv"
)

data.to_csv(
    output_path,
    index=False
)


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print("\n========================================")
print("       SYNTHETIC DATASET GENERATED")
print("========================================")

print(f"Samples : {len(data)}")
print(f"Features: {len(data.columns)}")

print("\nColumns:")
print(data.columns.tolist())

print("\nStress Level Distribution:")
print(
    data["stress_level"]
    .value_counts()
    .sort_index()
)

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset saved to:")
print(output_path)

print("========================================\n")