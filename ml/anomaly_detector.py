"""
Anomaly and Risk Detection Module

This module checks wearable physiological readings against
configured monitoring thresholds.

It is separate from the ML stress classification model.

Output:
    NORMAL
    WARNING
    HIGH RISK
"""

from typing import Dict, List


# ============================================================
# CONFIGURED MONITORING THRESHOLDS
# ============================================================

THRESHOLDS = {

    # Heart rate in BPM
    "hr_mean": {
        "min": 45.0,
        "max": 110.0,
        "name": "Heart Rate",
        "unit": "BPM",
    },

    # Electrodermal activity
    "eda_mean": {
        "min": 0.0,
        "max": 20.0,
        "name": "Electrodermal Activity",
        "unit": "µS",
    },

    # Skin temperature in Celsius
    "temp_mean": {
        "min": 25.0,
        "max": 40.0,
        "name": "Skin Temperature",
        "unit": "°C",
    },

    # Inter-beat interval in seconds
    "ibi_mean": {
        "min": 0.30,
        "max": 2.00,
        "name": "Inter-Beat Interval",
        "unit": "s",
    },

    # RMSSD in seconds
    "rmssd": {
        "min": 0.005,
        "max": 2.00,
        "name": "RMSSD",
        "unit": "s",
    },
}


def detect_anomaly(data: Dict) -> Dict:
    """
    Check wearable readings against configured thresholds.

    Parameters
    ----------
    data : Dict
        Dictionary containing wearable sensor features.

    Returns
    -------
    Dict
        Anomaly status, risk level and triggered alerts.
    """

    triggered_alerts: List[str] = []

    # ========================================================
    # CHECK EACH MONITORED FEATURE
    # ========================================================

    for feature, config in THRESHOLDS.items():

        value = data.get(feature)

        # Optional features such as IBI/RMSSD may be missing
        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        minimum = config["min"]
        maximum = config["max"]

        if value < minimum:

            triggered_alerts.append(
                f'{config["name"]} {value:.2f} {config["unit"]} '
                f'- below configured range'
            )

        elif value > maximum:

            triggered_alerts.append(
                f'{config["name"]} {value:.2f} {config["unit"]} '
                f'- above configured range'
            )

    # ========================================================
    # DETERMINE RISK LEVEL
    # ========================================================

    number_of_alerts = len(triggered_alerts)

    if number_of_alerts == 0:

        anomaly_status = "NORMAL"
        risk_level = "LOW"

    elif number_of_alerts == 1:

        anomaly_status = "ABNORMAL READING"
        risk_level = "WARNING"

    else:

        anomaly_status = "MULTIPLE ABNORMAL READINGS"
        risk_level = "HIGH"

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "anomaly_status": anomaly_status,
        "risk_level": risk_level,
        "triggered_alerts": triggered_alerts,
    }


# ============================================================
# SIMPLE LOCAL TEST
# ============================================================

if __name__ == "__main__":

    normal_data = {
        "hr_mean": 72,
        "eda_mean": 0.8,
        "temp_mean": 33.5,
        "ibi_mean": 0.85,
        "rmssd": 0.12,
    }

    abnormal_data = {
        "hr_mean": 118,
        "eda_mean": 1.2,
        "temp_mean": 33.8,
        "ibi_mean": 0.82,
        "rmssd": 0.10,
    }

    high_risk_data = {
        "hr_mean": 125,
        "eda_mean": 25,
        "temp_mean": 42,
        "ibi_mean": 0.20,
        "rmssd": 0.002,
    }

    print("\n========================================")
    print("      STRESSSENSE ANOMALY TEST")
    print("========================================")

    print("\nNORMAL DATA")
    print(detect_anomaly(normal_data))

    print("\nABNORMAL DATA")
    print(detect_anomaly(abnormal_data))

    print("\nHIGH RISK DATA")
    print(detect_anomaly(high_risk_data))
