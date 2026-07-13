"""Physics-grounded industrial sensor telemetry and Remaining Useful Life (RUL) generator."""

import math
from typing import Tuple
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG, PipelineConfig


def generate_industrial_telemetry(
    n_samples: int = 10000,
    seed: int = 42,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Generate multi-sensor industrial telemetry with physically grounded failure modes
    and continuous Remaining Useful Life (RUL) degradation trajectories.

    Failure Modes:
    - HDF (Heat Dissipation Failure): temp_diff < 8.6 K and RPM < 1380
    - PWF (Power Failure): mechanical_power outside [3500 W, 9000 W]
    - OSF (Overstrain Failure): tool_wear * torque > 11,000 min*Nm
    - TWF (Tool Wear Failure): tool_wear >= 200 min fatigue zone
    """
    rng = np.random.default_rng(seed)

    # 1. Base telemetry generation
    air_temp = np.clip(rng.normal(loc=300.0, scale=2.0, size=n_samples), 295.0, 305.0)
    process_temp = np.clip(air_temp + 10.0 + rng.normal(loc=0.0, scale=1.2, size=n_samples), 304.0, 316.0)
    rotational_speed = np.clip(rng.normal(loc=1538.0, scale=179.0, size=n_samples), 1100.0, 2800.0)
    torque = np.clip(rng.normal(loc=40.0, scale=9.9, size=n_samples), 10.0, 80.0)
    tool_wear = np.clip(rng.uniform(low=0.0, high=245.0, size=n_samples), 0.0, 250.0)

    # 2. Physics-grounded intermediate quantities
    temp_diff = process_temp - air_temp
    rad_per_sec = rotational_speed * (2.0 * math.pi / 60.0)
    mechanical_power = torque * rad_per_sec
    strain = tool_wear * torque

    # 3. Discrete failure conditions
    hdf = (temp_diff < config.temp_diff_critical_kelvin) & (rotational_speed < config.rpm_critical_low)
    pwf = (mechanical_power < config.nominal_power_min_watts) | (mechanical_power > config.nominal_power_max_watts)
    osf = strain > config.strain_critical_limit
    twf = (tool_wear >= config.tool_wear_critical_min) & (rng.uniform(0.0, 1.0, size=n_samples) < 0.45)

    failure_type = np.full(n_samples, "Normal", dtype=object)
    failure_type[hdf] = "Heat Dissipation Failure (HDF)"
    failure_type[pwf] = "Power Failure (PWF)"
    failure_type[osf] = "Overstrain Failure (OSF)"
    failure_type[twf] = "Tool Wear Failure (TWF)"

    machine_failure = (failure_type != "Normal").astype(int)

    # 4. Weibull-inspired Remaining Useful Life (RUL in hours)
    # RUL decays as tool wear approaches 250, strain approaches 11000, and thermal margin narrows
    wear_margin = np.clip(1.0 - (tool_wear / config.max_tool_life_min), 0.0, 1.0)
    strain_margin = np.clip(1.0 - (strain / (config.strain_critical_limit * 1.15)), 0.0, 1.0)
    thermal_margin = np.clip((temp_diff - 6.0) / 8.0, 0.05, 1.2)

    base_rul = 120.0 * (0.50 * wear_margin + 0.35 * strain_margin + 0.15 * thermal_margin)
    rul_noise = rng.normal(loc=0.0, scale=2.5, size=n_samples)
    rul_hours = np.clip(base_rul + rul_noise, 0.5, 120.0)
    # Failed machines have near-zero RUL
    rul_hours = np.where(machine_failure == 1, np.clip(rng.uniform(0.1, 3.5, size=n_samples), 0.1, 4.0), rul_hours)

    df = pd.DataFrame({
        "air_temperature_k": np.round(air_temp, 2),
        "process_temperature_k": np.round(process_temp, 2),
        "rotational_speed_rpm": np.round(rotational_speed, 1),
        "torque_nm": np.round(torque, 2),
        "tool_wear_min": np.round(tool_wear, 1),
        "rul_hours": np.round(rul_hours, 2),
        "failure_type": failure_type,
        "machine_failure": machine_failure,
    })
    return df


def load_train_test_split(
    n_samples: int = 10000,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving failure mode distributions."""
    data = generate_industrial_telemetry(n_samples=n_samples, seed=seed)

    normal = data[data["machine_failure"] == 0]
    failure = data[data["machine_failure"] == 1]

    train_normal = normal.sample(frac=1.0 - test_size, random_state=seed)
    test_normal = normal.drop(train_normal.index)

    train_fail = failure.sample(frac=1.0 - test_size, random_state=seed)
    test_fail = failure.drop(train_fail.index)

    train_df = pd.concat([train_normal, train_fail]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([test_normal, test_fail]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df
