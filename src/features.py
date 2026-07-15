"""Physics-informed and electromechanical feature engineering pipeline."""

import math
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import DEFAULT_CONFIG, PipelineConfig


class IndustrialFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms raw machine telemetry into non-linear thermodynamic and mechanical features.

    Engineered Features:
    1. temp_diff_k: Process Temp - Air Temp (Thermodynamic dissipation gradient)
    2. thermal_efficiency_index: temp_diff_k / (rotational_speed_rpm / 1000)
    3. angular_velocity_rad_s: RPM * (2 * pi / 60)
    4. mechanical_power_watts: Torque * Angular Velocity
    5. power_envelope_deviation: Distance outside nominal [3500W, 9000W] operating band
    6. strain_index: Tool Wear * Torque
    7. strain_load_ratio: strain_index / critical_strain_limit (11,000)
    8. wear_rate_per_rpm: Tool Wear / RPM
    9. torque_to_rpm_ratio: Torque / RPM
    10. kinetic_stress_proxy: Torque^2 * Angular Velocity / 1000
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_ = (
            self.config.feature_columns + self.config.engineered_columns
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        missing = [col for col in self.config.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required sensor columns in input: {missing}")

        air_temp = df["air_temperature_k"].to_numpy(dtype=float)
        proc_temp = df["process_temperature_k"].to_numpy(dtype=float)
        rpm = df["rotational_speed_rpm"].to_numpy(dtype=float)
        torque = df["torque_nm"].to_numpy(dtype=float)
        wear = df["tool_wear_min"].to_numpy(dtype=float)

        safe_rpm = np.where(rpm <= 0.0, 1.0, rpm)

        # 1. Thermodynamic gradient & efficiency
        temp_diff = proc_temp - air_temp
        df["temp_diff_k"] = temp_diff
        df["thermal_efficiency_index"] = temp_diff / (safe_rpm / 1000.0)

        # 2. Electromechanical kinematics & power
        omega = safe_rpm * (2.0 * math.pi / 60.0)
        df["angular_velocity_rad_s"] = omega
        power = torque * omega
        df["mechanical_power_watts"] = power

        # Deviation outside nominal power envelope [3500, 9000]
        low_dev = np.maximum(0.0, self.config.nominal_power_min_watts - power)
        high_dev = np.maximum(0.0, power - self.config.nominal_power_max_watts)
        df["power_envelope_deviation"] = low_dev + high_dev

        # 3. Structural strain & fatigue
        strain = torque * wear
        df["strain_index"] = strain
        df["strain_load_ratio"] = strain / self.config.strain_critical_limit
        df["wear_rate_per_rpm"] = wear / safe_rpm
        df["torque_to_rpm_ratio"] = torque / safe_rpm
        df["kinetic_stress_proxy"] = (torque ** 2) * omega / 1000.0

        output_df = df[self.feature_names_].fillna(0.0)
        return output_df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(self.feature_names_)
