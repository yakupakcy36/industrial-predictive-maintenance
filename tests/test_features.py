"""Test suite for feature extraction and physics-informed transformations."""

import math
import numpy as np
import pandas as pd
import pytest
from src.features import IndustrialFeatureTransformer
from src.config import DEFAULT_CONFIG


def test_feature_transformer_computations():
    """Verify physical formulas implemented in transformer."""
    transformer = IndustrialFeatureTransformer()
    
    sample_df = pd.DataFrame([{
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 100.0,
    }])
    
    transformed = transformer.fit_transform(sample_df)
    
    # Check thermodynamic gradient
    assert math.isclose(transformed["temp_diff_k"].iloc[0], 10.0, rel_tol=1e-3)
    
    # Check mechanical power: 40 * (1500 * 2 * pi / 60) = 40 * 157.0796 = ~6283.185 W
    expected_power = 40.0 * (1500.0 * 2.0 * math.pi / 60.0)
    assert math.isclose(transformed["mechanical_power_watts"].iloc[0], expected_power, rel_tol=1e-3)
    
    # Check strain index: 40 * 100 = 4000.0
    assert math.isclose(transformed["strain_index"].iloc[0], 4000.0, rel_tol=1e-3)
    
    # Check wear rate per rpm: 100 / 1500 = 0.06667
    assert math.isclose(transformed["wear_rate_per_rpm"].iloc[0], 100.0 / 1500.0, rel_tol=1e-3)


def test_feature_transformer_missing_column_raises():
    """Ensure missing telemetry inputs raise ValueError."""
    transformer = IndustrialFeatureTransformer()
    invalid_df = pd.DataFrame([{"air_temperature_k": 300.0}])
    
    with pytest.raises(ValueError, match="Missing required sensor columns"):
        transformer.fit_transform(invalid_df)


def test_feature_names_out():
    """Verify correct feature list exposed."""
    transformer = IndustrialFeatureTransformer()
    sample_df = pd.DataFrame([{
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 100.0,
    }])
    transformer.fit(sample_df)
    names = transformer.get_feature_names_out()
    assert "mechanical_power_watts" in names
    assert "strain_index" in names
    assert len(names) == len(DEFAULT_CONFIG.feature_columns) + len(DEFAULT_CONFIG.engineered_columns)
