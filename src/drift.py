"""Multi-method sensor telemetry drift & multivariate Mahalanobis anomaly detector."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from src.config import DEFAULT_CONFIG, PipelineConfig


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current telemetry."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct = (actual_counts / len(actual)) + eps

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_val, 4))


class SensorDriftDetector:
    """
    Evaluates univariate and multivariate sensor distribution shifts using:
    1. Two-Sample Kolmogorov-Smirnov (KS) Test
    2. Population Stability Index (PSI)
    3. First Wasserstein (Earth Mover's) Distance
    4. Multivariate Mahalanobis Covariance Distance
    """

    def __init__(self, baseline_df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config
        self.monitored_columns: List[str] = [
            col for col in config.feature_columns if col in baseline_df.columns
        ]
        self.baseline_summary: Dict[str, np.ndarray] = {
            col: baseline_df[col].dropna().to_numpy(dtype=float)
            for col in self.monitored_columns
        }

        # Precompute mean vector and regularized inverse covariance for Mahalanobis distance
        mat = baseline_df[self.monitored_columns].dropna().to_numpy(dtype=float)
        self.mean_vec = np.mean(mat, axis=0)
        cov = np.cov(mat, rowvar=False) + np.eye(len(self.monitored_columns)) * 1e-4
        self.inv_cov = np.linalg.inv(cov)

        # Baseline 99th percentile Mahalanobis threshold
        diffs = mat - self.mean_vec
        m_dists = np.sqrt(np.sum((diffs @ self.inv_cov) * diffs, axis=1))
        self.mahalanobis_threshold = float(np.percentile(m_dists, 99.0))

    def compute_mahalanobis(self, current_df: pd.DataFrame) -> np.ndarray:
        """Compute multivariate Mahalanobis distance for each row in current_df."""
        mat = current_df[self.monitored_columns].to_numpy(dtype=float)
        diffs = mat - self.mean_vec
        dists = np.sqrt(np.maximum(0.0, np.sum((diffs @ self.inv_cov) * diffs, axis=1)))
        return dists

    def evaluate(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate univariate and multivariate telemetry drift."""
        feature_reports: Dict[str, Dict[str, Any]] = {}
        drift_detected_count = 0

        for col in self.monitored_columns:
            if col not in current_df.columns or current_df[col].dropna().empty:
                continue

            ref_vals = self.baseline_summary[col]
            curr_vals = current_df[col].dropna().to_numpy(dtype=float)

            ks_res = stats.ks_2samp(ref_vals, curr_vals)
            p_val = float(ks_res.pvalue)
            stat = float(ks_res.statistic)

            psi = calculate_psi(ref_vals, curr_vals)
            w_dist = float(stats.wasserstein_distance(ref_vals, curr_vals))
            ref_std = max(1e-5, float(np.std(ref_vals)))
            normalized_wasserstein = w_dist / ref_std

            is_drift = (p_val < self.config.drift_ks_pvalue_threshold) or (
                psi >= self.config.psi_warning_threshold
            )

            if is_drift:
                drift_detected_count += 1

            severity = "NONE"
            if psi >= self.config.psi_critical_threshold or normalized_wasserstein > 0.75:
                severity = "CRITICAL"
            elif is_drift:
                severity = "WARNING"

            feature_reports[col] = {
                "ks_statistic": round(stat, 4),
                "ks_pvalue": round(p_val, 6),
                "psi": psi,
                "wasserstein_distance": round(w_dist, 4),
                "drift_severity": severity,
                "drift_detected": bool(is_drift),
                "current_mean": round(float(np.mean(curr_vals)), 2),
                "baseline_mean": round(float(np.mean(ref_vals)), 2),
            }

        # Multivariate Mahalanobis OOD check
        m_dists = self.compute_mahalanobis(current_df)
        ood_ratio = float(np.mean(m_dists > self.mahalanobis_threshold))

        overall_drift = (drift_detected_count > 0) or (ood_ratio > 0.15)
        return {
            "overall_drift_detected": bool(overall_drift),
            "drifted_features_count": drift_detected_count,
            "total_monitored_features": len(self.monitored_columns),
            "multivariate_mahalanobis_mean": round(float(np.mean(m_dists)), 4),
            "multivariate_ood_fraction": round(ood_ratio, 4),
            "feature_diagnostics": feature_reports,
        }
