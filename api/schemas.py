"""Pydantic v2 schemas for the Industrial Predictive Maintenance, RUL & XAI API."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class TelemetryReading(BaseModel):
    """Single machine telemetry snapshot."""
    air_temperature_k: float = Field(..., ge=250.0, le=350.0, description="Ambient air temperature in Kelvin")
    process_temperature_k: float = Field(..., ge=250.0, le=360.0, description="Machine process temperature in Kelvin")
    rotational_speed_rpm: float = Field(..., ge=500.0, le=4000.0, description="Spindle rotational speed in RPM")
    torque_nm: float = Field(..., ge=0.0, le=150.0, description="Spindle mechanical torque in Newton-meters")
    tool_wear_min: float = Field(..., ge=0.0, le=350.0, description="Tool cumulative operating wear in minutes")


class BatchTelemetryRequest(BaseModel):
    readings: List[TelemetryReading]


class PredictionResult(BaseModel):
    failure_probability: float
    estimated_rul_hours: float = Field(..., description="Remaining Useful Life estimate in operating hours")
    alert_level: str = Field(..., description="NORMAL, WARNING, or CRITICAL")
    primary_fault_mode: str = Field(..., description="Diagnosed failure mode (Normal, HDF, PWF, OSF, TWF)")
    recommended_action: str


class ExplanationResponse(BaseModel):
    prediction: PredictionResult
    engineered_telemetry: Dict[str, float]
    local_sensor_attributions: Dict[str, float]
    counterfactual_recommendation: str


class PredictionResponse(BaseModel):
    status: str = "success"
    predictions: List[PredictionResult]


class DriftEvaluationResponse(BaseModel):
    status: str = "success"
    overall_drift_detected: bool
    drifted_features_count: int
    total_monitored_features: int
    multivariate_mahalanobis_mean: float
    multivariate_ood_fraction: float
    feature_diagnostics: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    optimal_cost_threshold: float
    monitored_sensors: List[str]
