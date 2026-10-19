
from typing import Optional

from pydantic import BaseModel, Field


class WearableData(BaseModel):
    """
    Wearable sensor feature input expected by StressSense AI.
    These features match the public wearable stress dataset.
    """

    hr_mean: float = Field(..., description="Mean heart rate in BPM")
    hr_std: float = Field(..., description="Heart rate standard deviation")
    hr_min: float = Field(..., description="Minimum heart rate in BPM")
    hr_max: float = Field(..., description="Maximum heart rate in BPM")

    eda_mean: float = Field(..., description="Mean electrodermal activity")
    eda_std: float = Field(..., description="EDA standard deviation")
    eda_min: float = Field(..., description="Minimum EDA")
    eda_max: float = Field(..., description="Maximum EDA")

    temp_mean: float = Field(..., description="Mean skin temperature in Celsius")
    temp_std: float = Field(..., description="Skin temperature standard deviation")
    temp_min: float = Field(..., description="Minimum skin temperature in Celsius")
    temp_max: float = Field(..., description="Maximum skin temperature in Celsius")

    ibi_mean: Optional[float] = Field(
        None,
        description="Mean inter-beat interval in seconds"
    )

    ibi_std: Optional[float] = Field(
        None,
        description="Inter-beat interval standard deviation in seconds"
    )

    rmssd: Optional[float] = Field(
        None,
        description="RMSSD heart-rate variability feature in seconds"
    )

    acc_x_mean: float = Field(
        ...,
        description="Mean X-axis acceleration"
    )

    acc_y_mean: float = Field(
        ...,
        description="Mean Y-axis acceleration"
    )

    acc_z_mean: float = Field(
        ...,
        description="Mean Z-axis acceleration"
    )

    acc_magnitude_mean: float = Field(
        ...,
        description="Mean acceleration magnitude"
    )

    acc_magnitude_std: float = Field(
        ...,
        description="Acceleration magnitude standard deviation"
    )

    device_id: Optional[str] = Field(
        None,
        description="Optional wearable device identifier"
    )

    timestamp: Optional[str] = Field(
        None,
        description="Optional ISO timestamp"
    )


class StressPrediction(BaseModel):
    """
    Response returned by the stress prediction endpoint.
    """

    prediction: str
    stress_level: str
    predicted_class: int

    confidence: float

    rest_probability: float
    stress_probability: float

    anomaly_status: Optional[str] = None
    risk_level: Optional[str] = None
    triggered_alerts: Optional[list[str]] = None
