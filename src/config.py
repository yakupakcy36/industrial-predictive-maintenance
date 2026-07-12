"""Configuration and physical constants for Industrial Predictive Maintenance & RUL Engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PipelineConfig:
    """Master configuration for dataset, signal features, models, RUL, cost matrix, and drift."""
    random_seed: int = 42
    test_size: float = 0.2

    # Physical thresholds & thermodynamics constants
    nominal_power_min_watts: float = 3500.0
    nominal_power_max_watts: float = 9000.0
    temp_diff_critical_kelvin: float = 8.6
    rpm_critical_low: float = 1380.0
    strain_critical_limit: float = 11000.0
    tool_wear_critical_min: float = 200.0
    max_tool_life_min: float = 250.0

    # Base sensor telemetry columns
    feature_columns: List[str] = field(default_factory=lambda: [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ])

    # Physics & spectral engineered features
    engineered_columns: List[str] = field(default_factory=lambda: [
        "temp_diff_k",
        "thermal_efficiency_index",
        "mechanical_power_watts",
        "power_envelope_deviation",
        "strain_index",
        "strain_load_ratio",
        "wear_rate_per_rpm",
        "torque_to_rpm_ratio",
        "angular_velocity_rad_s",
        "kinetic_stress_proxy",
    ])

    # Asymmetric industrial cost parameters (USD)
    cost_false_positive_usd: float = 450.0     # Unplanned preventive inspection cost
    cost_false_negative_usd: float = 12500.0   # Catastrophic unplanned breakdown cost
    cost_true_positive_save_usd: float = 9500.0

    # Decision & Alert thresholds
    binary_alert_threshold: float = 0.50
    high_risk_threshold: float = 0.75
    drift_ks_pvalue_threshold: float = 0.05
    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.25

    # Artifact paths
    artifacts_dir: Path = Path("artifacts")
    model_filename: str = "predictive_maintenance_bundle.joblib"


DEFAULT_CONFIG = PipelineConfig()
