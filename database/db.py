import sqlite3
from pathlib import Path


# ============================================================
# DATABASE LOCATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_DIR = BASE_DIR / "data"

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_PATH = (
    DATABASE_DIR
    / "stress_predictions.db"
)


# ============================================================
# CONNECTION
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    # --------------------------------------------------------
    # Create table if it doesn't exist
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            device_id TEXT,
            timestamp TEXT,

            prediction TEXT NOT NULL,
            stress_level TEXT NOT NULL,
            predicted_class INTEGER NOT NULL,

            confidence REAL NOT NULL,
            rest_probability REAL NOT NULL,
            stress_probability REAL NOT NULL,

            anomaly_status TEXT,
            risk_level TEXT,
            triggered_alerts TEXT,

            hr_mean REAL,
            hr_std REAL,
            hr_min REAL,
            hr_max REAL,

            eda_mean REAL,
            eda_std REAL,
            eda_min REAL,
            eda_max REAL,

            temp_mean REAL,
            temp_std REAL,
            temp_min REAL,
            temp_max REAL,

            ibi_mean REAL,
            ibi_std REAL,
            rmssd REAL,

            acc_x_mean REAL,
            acc_y_mean REAL,
            acc_z_mean REAL,

            acc_magnitude_mean REAL,
            acc_magnitude_std REAL,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


    # --------------------------------------------------------
    # Migrate old database
    # --------------------------------------------------------

    cursor.execute(
        "PRAGMA table_info(predictions)"
    )

    existing_columns = {
        row["name"]
        for row in cursor.fetchall()
    }


    required_columns = {

        "device_id": "TEXT",

        "timestamp": "TEXT",

        "prediction": "TEXT",

        "stress_level": "TEXT",

        "predicted_class": "INTEGER",

        "confidence": "REAL",

        "rest_probability": "REAL",

        "stress_probability": "REAL",

        "anomaly_status": "TEXT",

        "risk_level": "TEXT",

        "triggered_alerts": "TEXT",

        "hr_mean": "REAL",
        "hr_std": "REAL",
        "hr_min": "REAL",
        "hr_max": "REAL",

        "eda_mean": "REAL",
        "eda_std": "REAL",
        "eda_min": "REAL",
        "eda_max": "REAL",

        "temp_mean": "REAL",
        "temp_std": "REAL",
        "temp_min": "REAL",
        "temp_max": "REAL",

        "ibi_mean": "REAL",
        "ibi_std": "REAL",
        "rmssd": "REAL",

        "acc_x_mean": "REAL",
        "acc_y_mean": "REAL",
        "acc_z_mean": "REAL",

        "acc_magnitude_mean": "REAL",
        "acc_magnitude_std": "REAL",
    }


    for column, column_type in required_columns.items():

        if column not in existing_columns:

            try:

                cursor.execute(
                    f"""
                    ALTER TABLE predictions
                    ADD COLUMN {column}
                    {column_type}
                    """
                )

                print(
                    f"✓ Added database column: {column}"
                )

            except sqlite3.OperationalError:

                pass


    connection.commit()

    connection.close()


    print(
        f"✓ Database initialized: {DATABASE_PATH}"
    )