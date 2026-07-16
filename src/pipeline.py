"""End-to-end execution pipeline and CLI for training, RUL evaluation, and XAI inference."""

import argparse
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.dataset import load_train_test_split
from src.model import train_models, PredictiveMaintenanceModelBundle


def run_training_pipeline(save_artifact: bool = True):
    """Executes data generation, tri-stage training, cost optimization, and serialization."""
    print("=" * 65)
    print("INDUSTRIAL PREDICTIVE MAINTENANCE & RUL ENGINE: TRAINING PIPELINE")
    print("=" * 65)

    print("[1/4] Generating & partitioning industrial sensor telemetry...")
    train_df, test_df = load_train_test_split(n_samples=10000, test_size=0.2)
    print(f"      Train samples: {len(train_df)} (Failures: {train_df['machine_failure'].sum()})")
    print(f"      Test samples:  {len(test_df)} (Failures: {test_df['machine_failure'].sum()})")

    print("[2/4] Fitting physics feature transformer, risk, fault, & RUL models...")
    bundle, metrics, diag_report = train_models(train_df, test_df, config=DEFAULT_CONFIG)

    print("\n[3/4] Out-of-Sample Performance & Cost Evaluation:")
    print("-" * 55)
    print(f"  ROC-AUC Score:               {metrics.roc_auc:.4f}")
    print(f"  PR-AUC Score:                {metrics.pr_auc:.4f}")
    print(f"  Binary Precision:            {metrics.precision:.4f}")
    print(f"  Binary Recall:               {metrics.recall:.4f}")
    print(f"  Binary F1-Score:             {metrics.f1:.4f}")
    print(f"  RUL Regression RMSE (hours): {metrics.rul_rmse_hours:.4f}")
    print(f"  RUL Regression R2 Score:     {metrics.rul_r2_score:.4f}")
    print(f"  Cost-Optimal Threshold:      {metrics.optimal_cost_threshold:.2f}")
    print(f"  Est. Test Fleet Savings:     ${metrics.estimated_cost_savings_usd:,.2f}")
    print("-" * 55)

    print("\nDiagnostic Multi-Class Accuracy Summary:")
    for fault, stats in diag_report.items():
        if isinstance(stats, dict) and "f1-score" in stats:
            print(f"  - {fault:32s} F1: {stats['f1-score']:.3f} | Support: {stats['support']}")

    if save_artifact:
        out_path = DEFAULT_CONFIG.artifacts_dir / DEFAULT_CONFIG.model_filename
        print(f"\n[4/4] Serializing model bundle to: {out_path}")
        bundle.save(str(out_path))
        print("      Serialization complete.")

    print("\n[SUCCESS] Pipeline execution finished successfully.")
    return bundle, metrics


def run_sample_inference():
    """Performs sample inference and XAI counterfactual analysis."""
    out_path = DEFAULT_CONFIG.artifacts_dir / DEFAULT_CONFIG.model_filename
    if not out_path.exists():
        bundle, _ = run_training_pipeline(save_artifact=True)
    else:
        bundle = PredictiveMaintenanceModelBundle.load(str(out_path))

    sample_data = pd.DataFrame([
        {
            "air_temperature_k": 300.1,
            "process_temperature_k": 310.2,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 35.0,
        },
        {
            "air_temperature_k": 301.0,
            "process_temperature_k": 311.5,
            "rotational_speed_rpm": 1250.0,
            "torque_nm": 68.0,
            "tool_wear_min": 215.0,
        },
    ])

    print("\nRunning Inference on Sample Batch:")
    predictions = bundle.predict(sample_data)
    print(pd.concat([sample_data, predictions], axis=1).to_string(index=False))

    print("\nRunning XAI Counterfactual Analysis on High-Risk Machine:")
    explanation = bundle.explain_instance(sample_data.iloc[1].to_dict())
    print("  Counterfactual:", explanation["counterfactual_recommendation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Predictive Maintenance CLI")
    parser.add_argument("command", choices=["train", "predict", "test"], help="Action to perform")
    args = parser.parse_args()

    if args.command == "train":
        run_training_pipeline(save_artifact=True)
    elif args.command == "predict":
        run_sample_inference()
    elif args.command == "test":
        print("Run `pytest` to execute test suite.")
