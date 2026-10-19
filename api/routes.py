from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    WearableData,
    StressPrediction,
)

from ml.predict import predict_stress
from ml.anomaly_detector import detect_anomaly

from database.models import (
    save_prediction,
    get_latest_prediction,
    get_prediction_history,
)


router = APIRouter()


# ============================================================
# ROOT
# ============================================================

@router.get("/")
def root():

    return {
        "status": "online",
        "service": "StressSense AI API",
        "model": "PyTorch Stress Classification Model",
        "classes": [
            "Rest",
            "Stress",
        ],
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# STRESS + ANOMALY PREDICTION
# ============================================================

@router.post(
    "/predict",
    response_model=StressPrediction,
)
def predict(
    data: WearableData,
):

    try:

        # ----------------------------------------------------
        # Convert input to dictionary
        # ----------------------------------------------------

        input_data = data.model_dump()

        # ----------------------------------------------------
        # 1. ML STRESS PREDICTION
        # ----------------------------------------------------

        stress_result = predict_stress(
            input_data
        )

        # ----------------------------------------------------
        # 2. ANOMALY / RISK DETECTION
        # ----------------------------------------------------

        anomaly_result = detect_anomaly(
            input_data
        )

        # ----------------------------------------------------
        # 3. COMBINE RESULTS
        # ----------------------------------------------------

        result = {
            **stress_result,
            **anomaly_result,
        }

        # ----------------------------------------------------
        # 4. SAVE RESULT TO DATABASE
        # ----------------------------------------------------

        save_prediction(

            device_id=input_data.get(
                "device_id"
            ),

            timestamp=input_data.get(
                "timestamp"
            ),

            prediction=result.get(
                "prediction"
            ),

            stress_level=result.get(
                "stress_level"
            ),

            predicted_class=result.get(
                "predicted_class"
            ),

            confidence=result.get(
                "confidence"
            ),

            rest_probability=result.get(
                "rest_probability"
            ),

            stress_probability=result.get(
                "stress_probability"
            ),

            anomaly_status=result.get(
                "anomaly_status"
            ),

            risk_level=result.get(
                "risk_level"
            ),

            triggered_alerts=result.get(
                "triggered_alerts"
            ),
            sensor_data=input_data,
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# LATEST PREDICTION
# ============================================================

@router.get("/latest")
def latest():

    result = get_latest_prediction()

    if result is None:

        raise HTTPException(
            status_code=404,
            detail="No predictions available yet.",
        )

    return result


# ============================================================
# PREDICTION HISTORY
# ============================================================

@router.get("/history")
def history(
    limit: int = Query(
        100,
        ge=1,
        le=1000,
    )
):

    results = get_prediction_history(
        limit
    )

    return {
        "count": len(results),
        "data": results,
    }