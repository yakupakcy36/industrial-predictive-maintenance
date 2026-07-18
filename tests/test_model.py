"""Test suite for tri-stage model training, RUL regression, XAI explanations, and cost optimization."""

import os
import pandas as pd
import pytest
from src.dataset import load_train_test_split
from src.model import train_models, PredictiveMaintenanceModelBundle
from src.config import DEFAULT_CONFIG


@pytest.fixture(scope="module")
def small_dataset():
    train_df, test_df = load_train_test_split(n_samples=1500, test_size=0.25, seed=42)
    return train_df, test_df


def test_model_training_rul_and_cost_metrics(small_dataset):
    """Verify tri-stage training outputs high ROC-AUC, RUL R2, and cost savings."""
    train_df, test_df = small_dataset
    bundle, metrics, _ = train_models(train_df, test_df, config=DEFAULT_CONFIG)

    assert 0.85 <= metrics.roc_auc <= 1.0, f"ROC-AUC {metrics.roc_auc} lower than expected"
    assert metrics.rul_r2_score >= 0.70, f"RUL R2 {metrics.rul_r2_score} lower than expected"
    assert metrics.rul_rmse_hours < 15.0
    assert 0.15 <= metrics.optimal_cost_threshold <= 0.85
    assert metrics.estimated_cost_savings_usd > 0.0

    predictions = bundle.predict(test_df)
    assert len(predictions) == len(test_df)
    assert "estimated_rul_hours" in predictions.columns
    assert (predictions["estimated_rul_hours"] > 0.0).all()


def test_xai_counterfactual_explanation(small_dataset):
    """Verify local feature attributions and counterfactual synthesis."""
    train_df, test_df = small_dataset
    bundle, _, _ = train_models(train_df, test_df, config=DEFAULT_CONFIG)

    high_risk_reading = {
        "air_temperature_k": 302.0,
        "process_temperature_k": 309.5,
        "rotational_speed_rpm": 1280.0,
        "torque_nm": 68.0,
        "tool_wear_min": 215.0,
    }
    explanation = bundle.explain_instance(high_risk_reading)
    assert "local_sensor_attributions" in explanation
    assert "counterfactual_recommendation" in explanation
    assert len(explanation["local_sensor_attributions"]) == 5


def test_bundle_save_and_load(tmp_path, small_dataset):
    """Verify bundle serialization integrity across all 3 stages."""
    train_df, test_df = small_dataset
    bundle, _, _ = train_models(train_df, test_df, config=DEFAULT_CONFIG)

    save_path = tmp_path / "test_model_bundle.joblib"
    bundle.save(str(save_path))
    assert os.path.exists(save_path)

    loaded_bundle = PredictiveMaintenanceModelBundle.load(str(save_path))
    preds_original = bundle.predict(test_df.head(10))
    preds_loaded = loaded_bundle.predict(test_df.head(10))

    pd.testing.assert_frame_equal(preds_original, preds_loaded)
