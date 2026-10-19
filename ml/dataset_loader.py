import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
)

WEARABLE_DIR = os.path.join(
    DATASET_DIR,
    "Wearable_Dataset"
)

STRESS_DIR = os.path.join(
    WEARABLE_DIR,
    "STRESS"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "stress_features.csv"
)


# ============================================================
# PROTOCOL DEFINITIONS
# ============================================================

V1_STAGES = [
    "Baseline",
    "Stroop",
    "First Rest",
    "TMCT",
    "Second Rest",
    "Real Opinion",
    "Opposite Opinion",
    "Subtract"
]

V2_STAGES = [
    "Baseline",
    "TMCT",
    "First Rest",
    "Real Opinion",
    "Opposite Opinion",
    "Second Rest",
    "Subtract"
]


# ============================================================
# ML CLASS DEFINITIONS
#
# 0 = Rest
# 1 = Stress
#
# These labels are derived from the experimental protocol.
# They are NOT medical diagnoses.
# ============================================================

REST_STAGES = {
    "Baseline",
    "First Rest",
    "Second Rest"
}

STRESS_STAGES = {
    "Stroop",
    "TMCT",
    "Real Opinion",
    "Opposite Opinion",
    "Subtract"
}


# ============================================================
# SENSOR LIST
# ============================================================

STANDARD_SENSORS = [
    "HR",
    "EDA",
    "TEMP",
    "ACC"
]


# ============================================================
# SAFE CSV READER
# ============================================================

def read_csv_safe(path):

    if not os.path.exists(path):
        return None

    try:

        if os.path.getsize(path) == 0:

            print(
                f"  Warning: empty file skipped: {path}"
            )

            return None

        data = pd.read_csv(
            path,
            header=None,
            encoding="latin1"
        )

        return data

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        OSError
    ) as error:

        print(
            f"  Warning: could not read {path}: {error}"
        )

        return None


# ============================================================
# STANDARD SENSOR LOADER
#
# File structure:
#
# Row 0 -> start timestamp
# Row 1 -> sampling frequency
# Row 2+ -> sensor values
# ============================================================

def load_sensor_data(
    subject_dir,
    sensor
):

    path = os.path.join(
        subject_dir,
        f"{sensor}.csv"
    )

    data = read_csv_safe(path)

    if data is None:
        return None

    if len(data) < 3:
        return None

    start_time = pd.to_datetime(
        data.iloc[0, 0],
        errors="coerce"
    )

    if pd.isna(start_time):
        return None

    frequency = pd.to_numeric(
        data.iloc[1, 0],
        errors="coerce"
    )

    if pd.isna(frequency):
        return None

    frequency = float(frequency)

    if frequency <= 0:
        return None

    values = data.iloc[2:].copy()

    return {
        "start_time": start_time,
        "frequency": frequency,
        "values": values
    }


# ============================================================
# IBI LOADER
#
# IMPORTANT:
#
# IBI.csv is NOT a normal sampled sensor.
#
# Column 0 = elapsed time in seconds
# Column 1 = IBI duration in seconds
#
# Therefore we do NOT use a sampling frequency.
# ============================================================

def load_ibi_data(subject_dir):

    path = os.path.join(
        subject_dir,
        "IBI.csv"
    )

    data = read_csv_safe(path)

    if data is None:
        return None

    if data.shape[1] < 2:
        return None

    if len(data) < 2:
        return None

    elapsed = pd.to_numeric(
        data.iloc[:, 0],
        errors="coerce"
    )

    ibi = pd.to_numeric(
        data.iloc[:, 1],
        errors="coerce"
    )

    result = pd.DataFrame({
        "elapsed_seconds": elapsed,
        "ibi_seconds": ibi
    })

    result = result.dropna()

    if result.empty:
        return None

    return result


# ============================================================
# TAG LOADER
# ============================================================

def load_tags(subject_dir):

    path = os.path.join(
        subject_dir,
        "tags.csv"
    )

    data = read_csv_safe(path)

    if data is None:
        return []

    if data.empty:
        return []

    tags = pd.to_datetime(
        data.iloc[:, 0],
        errors="coerce"
    )

    tags = tags.dropna()

    return tags.tolist()


# ============================================================
# CREATE TIMESTAMPS
# ============================================================

def create_timestamps(
    start_time,
    count,
    frequency
):

    if count <= 0:
        return pd.DatetimeIndex([])

    if frequency <= 0:
        return pd.DatetimeIndex([])

    interval = pd.to_timedelta(
        1.0 / frequency,
        unit="s"
    )

    return pd.date_range(
        start=start_time,
        periods=count,
        freq=interval
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

def clean_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()


# ============================================================
# SAFE STATISTICS
# ============================================================

def safe_mean(series):

    values = clean_numeric(series)

    if values.empty:
        return np.nan

    return float(
        values.mean()
    )


def safe_std(series):

    values = clean_numeric(series)

    if len(values) < 2:
        return 0.0

    return float(
        values.std()
    )


def safe_min(series):

    values = clean_numeric(series)

    if values.empty:
        return np.nan

    return float(
        values.min()
    )


def safe_max(series):

    values = clean_numeric(series)

    if values.empty:
        return np.nan

    return float(
        values.max()
    )


# ============================================================
# STANDARD SENSOR SEGMENT
# ============================================================

def get_sensor_segment(
    sensor_data,
    start,
    end
):

    if sensor_data is None:
        return pd.DataFrame()

    values = sensor_data["values"].copy()

    timestamps = create_timestamps(
        sensor_data["start_time"],
        len(values),
        sensor_data["frequency"]
    )

    if len(timestamps) != len(values):
        return pd.DataFrame()

    values.index = timestamps

    segment = values[
        (values.index >= start) &
        (values.index < end)
    ]

    return segment


# ============================================================
# IBI SEGMENT
# ============================================================

def get_ibi_segment(
    ibi_data,
    session_start,
    start,
    end
):

    if ibi_data is None:
        return pd.DataFrame()

    segment_start = (
        start - session_start
    ).total_seconds()

    segment_end = (
        end - session_start
    ).total_seconds()

    segment = ibi_data[
        (
            ibi_data["elapsed_seconds"]
            >= segment_start
        )
        &
        (
            ibi_data["elapsed_seconds"]
            < segment_end
        )
    ].copy()

    return segment


# ============================================================
# HRV FEATURES
# ============================================================

def calculate_hrv_features(
    ibi_segment
):

    if (
        ibi_segment is None
        or ibi_segment.empty
    ):

        return {
            "ibi_mean": np.nan,
            "ibi_std": np.nan,
            "rmssd": np.nan
        }

    ibi = clean_numeric(
        ibi_segment["ibi_seconds"]
    )

    if ibi.empty:

        return {
            "ibi_mean": np.nan,
            "ibi_std": np.nan,
            "rmssd": np.nan
        }

    differences = np.diff(
        ibi.values
    )

    if len(differences) == 0:

        rmssd = np.nan

    else:

        rmssd = float(
            np.sqrt(
                np.mean(
                    differences ** 2
                )
            )
        )

    return {
        "ibi_mean": float(
            ibi.mean()
        ),

        "ibi_std": (
            float(ibi.std())
            if len(ibi) > 1
            else 0.0
        ),

        "rmssd": rmssd
    }


# ============================================================
# ACCELEROMETER FEATURES
# ============================================================

def calculate_acc_features(
    acc_segment
):

    empty_result = {
        "acc_x_mean": np.nan,
        "acc_y_mean": np.nan,
        "acc_z_mean": np.nan,
        "acc_magnitude_mean": np.nan,
        "acc_magnitude_std": np.nan
    }

    if (
        acc_segment is None
        or acc_segment.empty
    ):

        return empty_result

    acc = acc_segment.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if acc.shape[1] < 3:
        return empty_result

    x = acc.iloc[:, 0]
    y = acc.iloc[:, 1]
    z = acc.iloc[:, 2]

    magnitude = np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )

    return {
        "acc_x_mean":
            safe_mean(x),

        "acc_y_mean":
            safe_mean(y),

        "acc_z_mean":
            safe_mean(z),

        "acc_magnitude_mean":
            safe_mean(magnitude),

        "acc_magnitude_std":
            safe_std(magnitude)
    }


# ============================================================
# STRESS LABEL
# ============================================================

def get_stress_label(stage):

    if stage in REST_STAGES:
        return "Rest"

    if stage in STRESS_STAGES:
        return "Stress"

    return np.nan


def get_stress_class(stage):

    label = get_stress_label(
        stage
    )

    if label == "Rest":
        return 0

    if label == "Stress":
        return 1

    return np.nan


# ============================================================
# LOAD STRESS SCORES
# ============================================================

def load_stress_scores(protocol):

    if protocol == "V1":

        filename = (
            "Stress_Level_v1.csv"
        )

    else:

        filename = (
            "Stress_Level_v2.csv"
        )

    path = os.path.join(
        DATASET_DIR,
        filename
    )

    data = pd.read_csv(
        path,
        encoding="latin1"
    )

    data = data.rename(
        columns={
            data.columns[0]:
                "subject_id"
        }
    )

    data["subject_id"] = (
        data["subject_id"]
        .astype(str)
        .str.strip()
    )

    return data.set_index(
        "subject_id"
    )


# ============================================================
# PROCESS ONE NORMAL SUBJECT
# ============================================================

def process_subject(
    subject_id,
    subject_dir
):

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    if subject_id.startswith("S"):

        protocol = "V1"
        stages = V1_STAGES

    elif subject_id.startswith("f"):

        protocol = "V2"
        stages = V2_STAGES

    else:

        print(
            f"  Unknown subject skipped: {subject_id}"
        )

        return []

    # --------------------------------------------------------
    # Tags
    # --------------------------------------------------------

    tags = load_tags(
        subject_dir
    )

    if len(tags) < 2:

        print(
            f"  Skipped {subject_id}: "
            f"not enough tags."
        )

        return []

    # --------------------------------------------------------
    # Stress scores
    # --------------------------------------------------------

    stress_scores = load_stress_scores(
        protocol
    )

    if subject_id not in stress_scores.index:

        print(
            f"  Skipped {subject_id}: "
            f"stress score not found."
        )

        return []

    subject_scores = (
        stress_scores.loc[
            subject_id
        ]
    )

    # --------------------------------------------------------
    # Sensors
    # --------------------------------------------------------

    hr_data = load_sensor_data(
        subject_dir,
        "HR"
    )

    eda_data = load_sensor_data(
        subject_dir,
        "EDA"
    )

    temp_data = load_sensor_data(
        subject_dir,
        "TEMP"
    )

    acc_data = load_sensor_data(
        subject_dir,
        "ACC"
    )

    ibi_data = load_ibi_data(
        subject_dir
    )

    # --------------------------------------------------------
    # Session start
    #
    # Prefer the earliest valid standard-sensor
    # timestamp.
    # --------------------------------------------------------

    valid_starts = []

    for sensor_data in [
        hr_data,
        eda_data,
        temp_data,
        acc_data
    ]:

        if sensor_data is not None:

            valid_starts.append(
                sensor_data["start_time"]
            )

    if not valid_starts:

        print(
            f"  Skipped {subject_id}: "
            f"no valid sensor data."
        )

        return []

    session_start = min(
        valid_starts
    )

    # --------------------------------------------------------
    # Number of usable stages
    # --------------------------------------------------------

    segment_count = min(
        len(stages),
        len(tags) - 1
    )

    rows = []

    # --------------------------------------------------------
    # Process every stage
    # --------------------------------------------------------

    for index in range(
        segment_count
    ):

        stage = stages[index]

        start = tags[index]

        end = tags[index + 1]

        if end <= start:
            continue

        # ----------------------------------------------------
        # Sensor segments
        # ----------------------------------------------------

        hr = get_sensor_segment(
            hr_data,
            start,
            end
        )

        eda = get_sensor_segment(
            eda_data,
            start,
            end
        )

        temp = get_sensor_segment(
            temp_data,
            start,
            end
        )

        acc = get_sensor_segment(
            acc_data,
            start,
            end
        )

        ibi = get_ibi_segment(
            ibi_data,
            session_start,
            start,
            end
        )

        # ----------------------------------------------------
        # Main feature row
        # ----------------------------------------------------

        row = {

            "subject_id":
                subject_id,

            "protocol":
                protocol,

            "stage":
                stage,

            "start_time":
                start,

            "end_time":
                end,

            # HR
            "hr_mean":
                (
                    safe_mean(
                        hr.iloc[:, 0]
                    )
                    if not hr.empty
                    else np.nan
                ),

            "hr_std":
                (
                    safe_std(
                        hr.iloc[:, 0]
                    )
                    if not hr.empty
                    else np.nan
                ),

            "hr_min":
                (
                    safe_min(
                        hr.iloc[:, 0]
                    )
                    if not hr.empty
                    else np.nan
                ),

            "hr_max":
                (
                    safe_max(
                        hr.iloc[:, 0]
                    )
                    if not hr.empty
                    else np.nan
                ),

            # EDA
            "eda_mean":
                (
                    safe_mean(
                        eda.iloc[:, 0]
                    )
                    if not eda.empty
                    else np.nan
                ),

            "eda_std":
                (
                    safe_std(
                        eda.iloc[:, 0]
                    )
                    if not eda.empty
                    else np.nan
                ),

            "eda_min":
                (
                    safe_min(
                        eda.iloc[:, 0]
                    )
                    if not eda.empty
                    else np.nan
                ),

            "eda_max":
                (
                    safe_max(
                        eda.iloc[:, 0]
                    )
                    if not eda.empty
                    else np.nan
                ),

            # Temperature
            "temp_mean":
                (
                    safe_mean(
                        temp.iloc[:, 0]
                    )
                    if not temp.empty
                    else np.nan
                ),

            "temp_std":
                (
                    safe_std(
                        temp.iloc[:, 0]
                    )
                    if not temp.empty
                    else np.nan
                ),

            "temp_min":
                (
                    safe_min(
                        temp.iloc[:, 0]
                    )
                    if not temp.empty
                    else np.nan
                ),

            "temp_max":
                (
                    safe_max(
                        temp.iloc[:, 0]
                    )
                    if not temp.empty
                    else np.nan
                )
        }

        # ----------------------------------------------------
        # HRV
        # ----------------------------------------------------

        row.update(
            calculate_hrv_features(
                ibi
            )
        )

        # ----------------------------------------------------
        # Accelerometer
        # ----------------------------------------------------

        row.update(
            calculate_acc_features(
                acc
            )
        )

        # ----------------------------------------------------
        # Original questionnaire score
        # ----------------------------------------------------

        if stage in subject_scores.index:

            stress_score = pd.to_numeric(
                subject_scores[stage],
                errors="coerce"
            )

        else:

            stress_score = np.nan

        row["stress_score"] = (
            stress_score
        )

        # ----------------------------------------------------
        # ML TARGET
        # ----------------------------------------------------

        row["stress_level"] = (
            get_stress_label(
                stage
            )
        )

        row["stress_class"] = (
            get_stress_class(
                stage
            )
        )

        rows.append(row)

    return rows


# ============================================================
# DISCOVER PARTICIPANTS
# ============================================================

def discover_participants():

    if not os.path.isdir(
        STRESS_DIR
    ):

        return []

    participants = {}

    for name in os.listdir(
        STRESS_DIR
    ):

        path = os.path.join(
            STRESS_DIR,
            name
        )

        if not os.path.isdir(path):
            continue

        # ----------------------------------------------------
        # Normal participant
        # ----------------------------------------------------

        if (
            name.startswith("S")
            and "_" not in name
        ):

            participants[name] = path

        # ----------------------------------------------------
        # V2 participant
        #
        # Normal:
        # f01, f02, ...
        #
        # Split:
        # f14_a
        # f14_b
        # ----------------------------------------------------

        elif name.startswith("f"):

            if name in [
                "f14_a",
                "f14_b"
            ]:

                continue

            participants[name] = path

    # --------------------------------------------------------
    # f14 handling
    #
    # f14_a has an empty tags.csv.
    # f14_b contains the complete tagged session.
    #
    # Therefore f14_b is used as the recording for
    # participant f14.
    # --------------------------------------------------------

    f14_b = os.path.join(
        STRESS_DIR,
        "f14_b"
    )

    if os.path.isdir(f14_b):

        tags = load_tags(
            f14_b
        )

        if len(tags) >= 2:

            participants["f14"] = f14_b

        else:

            print(
                "Warning: f14_b does not "
                "contain enough tags."
            )

    return sorted(
        participants.items()
    )


# ============================================================
# BUILD COMPLETE DATASET
# ============================================================

def build_dataset():

    print(
        "\n========================================"
    )

    print(
        "   STRESS DATASET PREPARATION"
    )

    print(
        "========================================"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Discover participants
    # --------------------------------------------------------

    participants = (
        discover_participants()
    )

    print(
        f"\nParticipants discovered: "
        f"{len(participants)}"
    )

    print(
        "Subjects:"
    )

    print(
        ", ".join(
            subject_id
            for subject_id, _
            in participants
        )
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    all_rows = []

    for subject_id, subject_dir in participants:

        print(
            f"\nProcessing {subject_id}..."
        )

        rows = process_subject(
            subject_id,
            subject_dir
        )

        if rows:

            print(
                f"  Extracted "
                f"{len(rows)} segments."
            )

        else:

            print(
                "  No usable segments extracted."
            )

        all_rows.extend(
            rows
        )

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if not all_rows:

        print(
            "\nERROR: No samples were generated."
        )

        return

    dataset = pd.DataFrame(
        all_rows
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    non_numeric = [
        "subject_id",
        "protocol",
        "stage",
        "start_time",
        "end_time",
        "stress_level"
    ]

    numerical_columns = [
        column
        for column in dataset.columns
        if column not in non_numeric
    ]

    for column in numerical_columns:

        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [

        # Heart rate
        "hr_mean",
        "hr_std",
        "hr_min",
        "hr_max",

        # Electrodermal activity
        "eda_mean",
        "eda_std",
        "eda_min",
        "eda_max",

        # Temperature
        "temp_mean",
        "temp_std",
        "temp_min",
        "temp_max",

        # HRV / IBI
        "ibi_mean",
        "ibi_std",
        "rmssd",

        # Accelerometer
        "acc_x_mean",
        "acc_y_mean",
        "acc_z_mean",
        "acc_magnitude_mean",
        "acc_magnitude_std"
    ]

    # --------------------------------------------------------
    # Remove rows where ALL physiological features
    # are missing.
    # --------------------------------------------------------

    dataset = dataset.dropna(
        subset=feature_columns,
        how="all"
    )

    # --------------------------------------------------------
    # Remove missing target rows
    # --------------------------------------------------------

    dataset = dataset.dropna(
        subset=[
            "stress_class"
        ]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    dataset = dataset.sort_values(
        by=[
            "subject_id",
            "start_time"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "       DATASET CREATED"
    )

    print(
        "========================================"
    )

    print(
        f"Total samples        : "
        f"{len(dataset)}"
    )

    print(
        f"Subjects represented : "
        f"{dataset['subject_id'].nunique()}"
    )

    print(
        f"Features extracted   : "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print(
        "\nClass distribution:"
    )

    print(
        dataset[
            "stress_level"
        ].value_counts()
    )

    print(
        "\nClass percentages:"
    )

    print(
        (
            dataset[
                "stress_level"
            ]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )

    # --------------------------------------------------------
    # Original questionnaire scores
    # --------------------------------------------------------

    print(
        "\nOriginal self-reported stress scores:"
    )

    print(
        dataset[
            "stress_score"
        ].describe()
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print(
        "\nMissing values:"
    )

    missing = (
        dataset[
            feature_columns
        ].isna().sum()
    )

    if (
        missing > 0
    ).any():

        print(
            missing[
                missing > 0
            ]
        )

    else:

        print(
            "No missing feature values."
        )

    # --------------------------------------------------------
    # Subject distribution
    # --------------------------------------------------------

    print(
        "\nSamples per subject:"
    )

    print(
        dataset[
            "subject_id"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Stage distribution
    # --------------------------------------------------------

    print(
        "\nSamples per stage:"
    )

    print(
        dataset[
            "stage"
        ].value_counts()
    )

    # --------------------------------------------------------
    # f14 confirmation
    # --------------------------------------------------------

    if "f14" in set(
        dataset["subject_id"]
    ):

        print(
            "\n✓ f14 successfully included "
            "using f14_b recording."
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print(
        "\nFirst 10 samples:"
    )

    print(
        dataset.head(10).to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_dataset()