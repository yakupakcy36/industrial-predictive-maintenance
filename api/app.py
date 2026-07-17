"""FastAPI application for Industrial Predictive Maintenance, RUL Estimation, and XAI."""

from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.config import DEFAULT_CONFIG
from src.model import PredictiveMaintenanceModelBundle
from src.pipeline import run_training_pipeline
from api.schemas import (
    TelemetryReading,
    BatchTelemetryRequest,
    PredictionResult,
    ExplanationResponse,
    PredictionResponse,
    DriftEvaluationResponse,
    HealthResponse,
)

bundle: Optional[PredictiveMaintenanceModelBundle] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring tri-stage model bundle is loaded or trained on startup."""
    global bundle
    artifact_path = DEFAULT_CONFIG.artifacts_dir / DEFAULT_CONFIG.model_filename

    if artifact_path.exists():
        try:
            bundle = PredictiveMaintenanceModelBundle.load(str(artifact_path))
            if not hasattr(bundle, "rul_pipeline"):
                bundle, _ = run_training_pipeline(save_artifact=True)
        except Exception:
            bundle, _ = run_training_pipeline(save_artifact=True)
    else:
        bundle, _ = run_training_pipeline(save_artifact=True)

    yield


app = FastAPI(
    title="Industrial Predictive Maintenance, RUL & XAI Diagnostics API",
    description=(
        "Production REST API for real-time equipment failure probability, Remaining Useful Life (RUL) "
        "prognostics, root-cause fault diagnostics (HDF, PWF, OSF, TWF), counterfactual XAI explanations, "
        "and multivariate Mahalanobis drift monitoring."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=(bundle is not None),
        version="2.0.0",
        optimal_cost_threshold=bundle.optimal_threshold if bundle else DEFAULT_CONFIG.binary_alert_threshold,
        monitored_sensors=DEFAULT_CONFIG.feature_columns,
    )


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict_single(reading: TelemetryReading):
    """Score a single machine telemetry reading for failure risk, RUL hours, and fault mode."""
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive maintenance model is not loaded.",
        )

    df = pd.DataFrame([reading.model_dump()])
    res = bundle.predict(df).iloc[0].to_dict()
    return PredictionResult(**res)


@app.post("/predict/explain", response_model=ExplanationResponse, tags=["Explainable AI (XAI)"])
def explain_prediction(reading: TelemetryReading):
    """Compute local sensor attributions and counterfactual operational recommendations."""
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive maintenance model is not loaded.",
        )

    exp = bundle.explain_instance(reading.model_dump())
    return ExplanationResponse(
        prediction=PredictionResult(**exp["prediction"]),
        engineered_telemetry=exp["engineered_telemetry"],
        local_sensor_attributions=exp["local_sensor_attributions"],
        counterfactual_recommendation=exp["counterfactual_recommendation"],
    )


@app.post("/predict/batch", response_model=PredictionResponse, tags=["Inference"])
def predict_batch(request: BatchTelemetryRequest):
    """Score a batch of sensor readings across fleet equipment."""
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive maintenance model is not loaded.",
        )

    if not request.readings:
        return PredictionResponse(predictions=[])

    df = pd.DataFrame([r.model_dump() for r in request.readings])
    results_df = bundle.predict(df)
    results = [PredictionResult(**row) for row in results_df.to_dict(orient="records")]
    return PredictionResponse(predictions=results)


@app.post("/drift/evaluate", response_model=DriftEvaluationResponse, tags=["Monitoring"])
def evaluate_drift(request: BatchTelemetryRequest):
    """
    Evaluate univariate (KS-test, PSI, Wasserstein) and multivariate (Mahalanobis)
    sensor distribution drift against baseline telemetry.
    """
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive maintenance model is not loaded.",
        )

    if len(request.readings) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Minimum 10 telemetry samples required for statistical drift analysis.",
        )

    df = pd.DataFrame([r.model_dump() for r in request.readings])
    report = bundle.drift_detector.evaluate(df)
    return DriftEvaluationResponse(status="success", **report)
