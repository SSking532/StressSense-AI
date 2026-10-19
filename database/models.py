from database.db import get_connection


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    device_id,
    timestamp,
    prediction,
    stress_level,
    predicted_class,
    confidence,
    rest_probability,
    stress_probability,
    anomaly_status,
    risk_level,
    triggered_alerts,
    sensor_data=None,
):

    connection = get_connection()

    cursor = connection.cursor()


    # --------------------------------------------------------
    # Alerts
    # --------------------------------------------------------

    alerts_text = ""

    if triggered_alerts:

        alerts_text = " | ".join(
            triggered_alerts
        )


    # --------------------------------------------------------
    # Sensor values
    # --------------------------------------------------------

    sensor_data = sensor_data or {}


    # --------------------------------------------------------
    # Insert
    # --------------------------------------------------------

    cursor.execute(
        """
        INSERT INTO predictions (

            device_id,
            timestamp,

            prediction,
            stress_level,
            predicted_class,

            confidence,
            rest_probability,
            stress_probability,

            anomaly_status,
            risk_level,
            triggered_alerts,

            hr_mean,
            hr_std,
            hr_min,
            hr_max,

            eda_mean,
            eda_std,
            eda_min,
            eda_max,

            temp_mean,
            temp_std,
            temp_min,
            temp_max,

            ibi_mean,
            ibi_std,
            rmssd,

            acc_x_mean,
            acc_y_mean,
            acc_z_mean,

            acc_magnitude_mean,
            acc_magnitude_std

        )

        VALUES (

            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,

            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?

        )
        """,

        (

            device_id,
            timestamp,

            prediction,
            stress_level,
            predicted_class,

            confidence,
            rest_probability,
            stress_probability,

            anomaly_status,
            risk_level,
            alerts_text,


            # Heart rate

            sensor_data.get(
                "hr_mean"
            ),

            sensor_data.get(
                "hr_std"
            ),

            sensor_data.get(
                "hr_min"
            ),

            sensor_data.get(
                "hr_max"
            ),


            # EDA

            sensor_data.get(
                "eda_mean"
            ),

            sensor_data.get(
                "eda_std"
            ),

            sensor_data.get(
                "eda_min"
            ),

            sensor_data.get(
                "eda_max"
            ),


            # Temperature

            sensor_data.get(
                "temp_mean"
            ),

            sensor_data.get(
                "temp_std"
            ),

            sensor_data.get(
                "temp_min"
            ),

            sensor_data.get(
                "temp_max"
            ),


            # IBI / HRV

            sensor_data.get(
                "ibi_mean"
            ),

            sensor_data.get(
                "ibi_std"
            ),

            sensor_data.get(
                "rmssd"
            ),


            # Acceleration

            sensor_data.get(
                "acc_x_mean"
            ),

            sensor_data.get(
                "acc_y_mean"
            ),

            sensor_data.get(
                "acc_z_mean"
            ),

            sensor_data.get(
                "acc_magnitude_mean"
            ),

            sensor_data.get(
                "acc_magnitude_std"
            ),
        )
    )


    connection.commit()

    prediction_id = cursor.lastrowid

    connection.close()

    return prediction_id


# ============================================================
# LATEST
# ============================================================

def get_latest_prediction():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM predictions

        ORDER BY id DESC

        LIMIT 1
        """
    )


    row = cursor.fetchone()

    connection.close()


    if row is None:

        return None


    return dict(row)


# ============================================================
# HISTORY
# ============================================================

def get_prediction_history(
    limit=100
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM predictions

        ORDER BY id DESC

        LIMIT ?
        """,
        (limit,)
    )


    rows = cursor.fetchall()

    connection.close()


    return [
        dict(row)
        for row in rows
    ]