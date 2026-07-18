"""Test suite for sensor telemetry drift and PSI calculation."""

import numpy as np
import pandas as pd
from src.drift import calculate_psi, SensorDriftDetector
from src.dataset import generate_industrial_telemetry


def test_psi_identical_distribution():
    """Identical distributions should yield near-zero PSI."""
    np.random.seed(42)
    dist = np.random.normal(loc=100.0, scale=15.0, size=2000)
    psi = calculate_psi(dist, dist)
    assert psi < 0.05, f"Expected low PSI for identical data, got {psi}"


def test_psi_shifted_distribution():
    """Heavily shifted distribution should yield high PSI."""
    np.random.seed(42)
    ref = np.random.normal(loc=100.0, scale=10.0, size=2000)
    shifted = np.random.normal(loc=130.0, scale=10.0, size=2000)
    psi = calculate_psi(ref, shifted)
    assert psi >= 0.25, f"Expected high PSI for shifted data, got {psi}"


def test_sensor_drift_detector():
    """Verify detector flags significant distribution drift."""
    base_df = generate_industrial_telemetry(n_samples=1500, seed=42)
    detector = SensorDriftDetector(baseline_df=base_df)

    # In-distribution sample
    clean_sample = generate_industrial_telemetry(n_samples=500, seed=99)
    clean_report = detector.evaluate(clean_sample)
    assert clean_report["overall_drift_detected"] is False

    # Artificially corrupted/drifted sensor telemetry (severe temperature sensor drift)
    corrupted_sample = clean_sample.copy()
    corrupted_sample["air_temperature_k"] = corrupted_sample["air_temperature_k"] + 15.0
    drift_report = detector.evaluate(corrupted_sample)
    assert drift_report["overall_drift_detected"] is True
    assert drift_report["feature_diagnostics"]["air_temperature_k"]["drift_detected"] is True
