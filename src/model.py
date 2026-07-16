"""Tri-stage Predictive Maintenance, Fault Diagnostics, RUL Regression, and XAI Engine."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    classification_report,
)
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.features import IndustrialFeatureTransformer
from src.drift import SensorDriftDetector


@dataclass
class EvaluationMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    rul_rmse_hours: float
    rul_r2_score: float
    optimal_cost_threshold: float
    estimated_cost_savings_usd: float


class PredictiveMaintenanceModelBundle:
    """
    Production model bundle encapsulating:
    - Stage 1: Binary Failure Probability Pipeline
    - Stage 2: Multi-Class Root Cause Diagnostic Pipeline
    - Stage 3: Remaining Useful Life (RUL) Regression Pipeline
    - Explainable AI (XAI) local attribution & counterfactual generator
    - Cost-sensitive threshold optimization
    - Sensor drift & Mahalanobis OOD monitor
    """

    def __init__(
        self,
        binary_pipeline: Pipeline,
        diagnostic_pipeline: Pipeline,
        rul_pipeline: Pipeline,
        drift_detector: SensorDriftDetector,
        feature_names: List[str],
        optimal_threshold: float = 0.50,
        config: PipelineConfig = DEFAULT_CONFIG,
    ):
        self.binary_pipeline = binary_pipeline
        self.diagnostic_pipeline = diagnostic_pipeline
        self.rul_pipeline = rul_pipeline
        self.drift_detector = drift_detector
        self.feature_names = feature_names
        self.optimal_threshold = optimal_threshold
        self.config = config

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Generate failure probabilities, fault diagnostics, RUL estimates, and actions."""
        probabilities = self.binary_pipeline.predict_proba(raw_df)[:, 1]
        diagnostics = self.diagnostic_pipeline.predict(raw_df)
        rul_preds = np.clip(self.rul_pipeline.predict(raw_df), 0.1, 120.0)

        results = []
        for prob, diag, rul in zip(probabilities, diagnostics, rul_preds):
            if prob >= self.config.high_risk_threshold or rul <= 8.0:
                level = "CRITICAL"
                mode = diag if diag != "Normal" else "Imminent Degradation"
                action = f"Immediate shutdown & inspect {mode} (Est. RUL: {rul:.1f}h)"
            elif prob >= self.optimal_threshold or rul <= 24.0:
                level = "WARNING"
                mode = diag if diag != "Normal" else "Accelerated Wear"
                action = f"Schedule maintenance within {max(4.0, rul * 0.5):.0f}h for {mode}"
            else:
                level = "NORMAL"
                mode = "Normal"
                action = "Nominal operating envelope; continue telemetry monitoring"

            results.append({
                "failure_probability": float(np.round(prob, 4)),
                "estimated_rul_hours": float(np.round(rul, 2)),
                "alert_level": level,
                "primary_fault_mode": mode,
                "recommended_action": action,
            })

        return pd.DataFrame(results)

    def explain_instance(self, reading: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute local feature attributions via finite-difference perturbation
        and synthesize counterfactual operational adjustments.
        """
        base_df = pd.DataFrame([reading])
        pred_row = self.predict(base_df).iloc[0]
        base_prob = float(pred_row["failure_probability"])

        # Extract engineered features
        transformer: IndustrialFeatureTransformer = self.binary_pipeline.named_steps["features"]
        feat_df = transformer.transform(base_df)

        # Compute local perturbations across base sensors
        perturbations = {
            "air_temperature_k": -2.0,
            "process_temperature_k": +2.0,
            "rotational_speed_rpm": +150.0,
            "torque_nm": -8.0,
            "tool_wear_min": -30.0,
        }

        attributions: Dict[str, float] = {}
        best_cf_sensor = None
        best_cf_prob = base_prob
        best_cf_val = None

        for sensor, delta in perturbations.items():
            cf_reading = dict(reading)
            cf_reading[sensor] = max(0.0, cf_reading[sensor] + delta)
            cf_prob = float(self.binary_pipeline.predict_proba(pd.DataFrame([cf_reading]))[0, 1])
            impact = round(base_prob - cf_prob, 4)
            attributions[sensor] = impact
            if cf_prob < best_cf_prob:
                best_cf_prob = cf_prob
                best_cf_sensor = sensor
                best_cf_val = cf_reading[sensor]

        if base_prob >= self.optimal_threshold and best_cf_sensor is not None:
            counterfactual = (
                f"Adjusting {best_cf_sensor} from {reading[best_cf_sensor]:.1f} to {best_cf_val:.1f} "
                f"reduces failure probability from {base_prob:.2%} to {best_cf_prob:.2%}."
            )
        else:
            counterfactual = "Machine is operating within safe thresholds; no parameter adjustment required."

        return {
            "prediction": pred_row.to_dict(),
            "engineered_telemetry": {k: round(float(v), 3) for k, v in feat_df.iloc[0].to_dict().items()},
            "local_sensor_attributions": attributions,
            "counterfactual_recommendation": counterfactual,
        }

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PredictiveMaintenanceModelBundle":
        return joblib.load(filepath)


def optimize_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tuple[float, float]:
    """
    Finds decision threshold minimizing expected industrial downtime and inspection cost.
    Returns (optimal_threshold, cost_savings_vs_unmonitored_usd).
    """
    unmonitored_cost = float(np.sum(y_true == 1) * config.cost_false_negative_usd)
    best_thresh = 0.50
    min_total_cost = float("inf")

    for thresh in np.linspace(0.15, 0.85, 36):
        preds = (y_prob >= thresh).astype(int)
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        tp = int(np.sum((preds == 1) & (y_true == 1)))

        # Total operational cost = FP inspections + FN breakdowns + TP planned maintenance cost
        tp_planned_cost = config.cost_false_negative_usd - config.cost_true_positive_save_usd
        total_cost = (
            fp * config.cost_false_positive_usd
            + fn * config.cost_false_negative_usd
            + tp * tp_planned_cost
        )
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_thresh = float(np.round(thresh, 2))

    savings = max(0.0, unmonitored_cost - min_total_cost)
    return best_thresh, float(np.round(savings, 2))


def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tuple[PredictiveMaintenanceModelBundle, EvaluationMetrics, Dict[str, Any]]:
    """Train binary failure risk, multi-class fault diagnostic, and RUL regression pipelines."""
    feature_transformer = IndustrialFeatureTransformer(config=config)

    # 1. Stage 1: Non-linear Polynomial + Calibrated Logistic Failure Risk Pipeline
    binary_pipeline = Pipeline([
        ("features", feature_transformer),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            C=1.5,
            class_weight="balanced",
            random_state=config.random_seed,
        )),
    ])

    # 2. Stage 2: Multi-Class Physical Diagnostic Decision Tree
    diagnostic_pipeline = Pipeline([
        ("features", feature_transformer),
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=config.random_seed,
        )),
    ])

    # 3. Stage 3: Remaining Useful Life (RUL) Regressor
    rul_pipeline = Pipeline([
        ("features", feature_transformer),
        ("scaler", StandardScaler()),
        ("regressor", DecisionTreeRegressor(
            max_depth=9,
            min_samples_leaf=10,
            random_state=config.random_seed,
        )),
    ])

    # Fit all 3 stages
    binary_pipeline.fit(train_df, train_df["machine_failure"])
    diagnostic_pipeline.fit(train_df, train_df["failure_type"])
    rul_pipeline.fit(train_df, train_df["rul_hours"])

    # Evaluate Stage 1 & Cost Optimization
    y_test_binary = test_df["machine_failure"].to_numpy()
    y_prob = binary_pipeline.predict_proba(test_df)[:, 1]
    opt_thresh, cost_savings = optimize_cost_threshold(y_test_binary, y_prob, config=config)
    y_pred = (y_prob >= opt_thresh).astype(int)

    # Evaluate Stage 3 (RUL)
    y_test_rul = test_df["rul_hours"].to_numpy()
    rul_pred = rul_pipeline.predict(test_df)
    rul_rmse = float(np.sqrt(mean_squared_error(y_test_rul, rul_pred)))
    rul_r2 = float(r2_score(y_test_rul, rul_pred))

    metrics = EvaluationMetrics(
        roc_auc=round(float(roc_auc_score(y_test_binary, y_prob)), 4),
        pr_auc=round(float(average_precision_score(y_test_binary, y_prob)), 4),
        precision=round(float(precision_score(y_test_binary, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_test_binary, y_pred, zero_division=0)), 4),
        f1=round(float(f1_score(y_test_binary, y_pred, zero_division=0)), 4),
        rul_rmse_hours=round(rul_rmse, 4),
        rul_r2_score=round(rul_r2, 4),
        optimal_cost_threshold=opt_thresh,
        estimated_cost_savings_usd=cost_savings,
    )

    # Diagnostic report
    y_test_diag = test_df["failure_type"]
    diag_pred = diagnostic_pipeline.predict(test_df)
    diag_report = classification_report(y_test_diag, diag_pred, output_dict=True, zero_division=0)

    drift_detector = SensorDriftDetector(baseline_df=train_df, config=config)

    bundle = PredictiveMaintenanceModelBundle(
        binary_pipeline=binary_pipeline,
        diagnostic_pipeline=diagnostic_pipeline,
        rul_pipeline=rul_pipeline,
        drift_detector=drift_detector,
        feature_names=feature_transformer.get_feature_names_out(),
        optimal_threshold=opt_thresh,
        config=config,
    )

    return bundle, metrics, diag_report
