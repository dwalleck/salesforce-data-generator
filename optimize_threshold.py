"""Find optimal prediction threshold for churn models.

For imbalanced classification problems, the default 0.5 threshold often
performs poorly. This script helps find a better threshold by evaluating
precision-recall trade-offs.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from model_trainer import prepare_features, time_based_split


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
    min_precision: float = 0.5,
) -> tuple[float, dict]:
    """
    Find optimal prediction threshold for imbalanced classification.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'recall', or 'custom')
        min_precision: Minimum acceptable precision (for 'custom' metric)

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    if metric == "f1":
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
    elif metric == "recall":
        # Find threshold with best recall while maintaining min_precision
        valid_idx = precisions >= min_precision
        if not valid_idx.any():
            print(f"Warning: No thresholds achieve precision >= {min_precision}")
            print(f"Using best F1 score instead")
            best_idx = np.argmax(f1_scores)
        else:
            # Among valid thresholds, choose one with best recall
            valid_recalls = recalls.copy()
            valid_recalls[~valid_idx] = 0
            best_idx = np.argmax(valid_recalls)
    else:  # custom
        # Maximize recall while maintaining min_precision
        valid_idx = precisions >= min_precision
        if not valid_idx.any():
            best_idx = np.argmax(f1_scores)
        else:
            valid_f1 = f1_scores.copy()
            valid_f1[~valid_idx] = 0
            best_idx = np.argmax(valid_f1)

    # Handle edge case where best_idx might be beyond thresholds array
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1

    optimal_threshold = thresholds[best_idx]

    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

    metrics = {
        "threshold": float(optimal_threshold),
        "precision": float(precision_score(y_true, y_pred_optimal, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_optimal, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred_optimal, zero_division=0)),
        "predicted_churn_rate": float(y_pred_optimal.mean()),
    }

    return optimal_threshold, metrics


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    optimal_threshold: float,
    output_path: Path | str | None = None,
) -> None:
    """
    Plot precision, recall, and F1 score vs threshold.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        optimal_threshold: Optimal threshold to highlight
        output_path: Path to save plot (optional)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Thresholds array is 1 element shorter, so trim precision/recall
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    f1_scores = f1_scores[:-1]

    plt.figure(figsize=(12, 6))

    plt.plot(thresholds, precisions, label="Precision", linewidth=2)
    plt.plot(thresholds, recalls, label="Recall", linewidth=2)
    plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2)

    # Highlight optimal threshold
    plt.axvline(
        optimal_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Optimal Threshold ({optimal_threshold:.3f})",
    )

    # Default threshold for reference
    plt.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default (0.5)")

    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Precision, Recall, and F1 Score vs Prediction Threshold", fontsize=14)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved threshold analysis: {output_path}")
    else:
        plt.show()

    plt.close()


def main() -> int:
    """Main entry point for threshold optimization."""
    parser = argparse.ArgumentParser(
        description="Find optimal prediction threshold for churn models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find optimal threshold for XGBoost model
  python optimize_threshold.py --input data/features.csv --model models_3month/xgboost.joblib

  # Optimize for recall with minimum 50% precision
  python optimize_threshold.py --input data/features.csv \\
      --model models_3month/xgboost.joblib --metric recall --min-precision 0.5

  # Compare all models
  python optimize_threshold.py --input data/features.csv --compare-all
        """,
    )

    parser.add_argument("--input", type=str, required=True, help="Path to feature-engineered CSV file")

    parser.add_argument("--model", type=str, help="Path to trained model (.joblib)")

    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare thresholds for all models in models_3month/ directory",
    )

    parser.add_argument(
        "--metric",
        choices=["f1", "recall", "custom"],
        default="f1",
        help="Metric to optimize (default: f1)",
    )

    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.5,
        help="Minimum acceptable precision (for recall/custom optimization, default: 0.5)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_3month",
        help="Directory to save plots (default: results_3month/)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Use same time-based split as training
    print("Splitting data...")
    _, _, test_df = time_based_split(df, train_months=9, val_months=2, test_months=1)

    X_test, y_test = prepare_features(test_df)
    print(f"  Test set: {len(X_test)} samples, {y_test.sum()} churned ({y_test.mean():.1%})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.compare_all:
        # Compare all models
        models_dir = Path("models_3month")
        model_files = list(models_dir.glob("*.joblib"))

        if not model_files:
            print("Error: No models found in models_3month/ directory")
            return 1

        print(f"\nComparing {len(model_files)} models...")
        results = {}

        for model_path in model_files:
            model_name = model_path.stem
            print(f"\n{model_name}:")

            model = joblib.load(model_path)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            optimal_threshold, metrics = find_optimal_threshold(
                y_test, y_pred_proba, metric=args.metric, min_precision=args.min_precision
            )

            results[model_name] = metrics

            print(f"  Optimal threshold: {metrics['threshold']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1_score']:.3f}")
            print(f"  Predicted churn rate: {metrics['predicted_churn_rate']:.1%}")

            # Plot threshold analysis
            plot_path = output_dir / f"{model_name}_threshold_analysis.png"
            plot_threshold_analysis(y_test, y_pred_proba, optimal_threshold, plot_path)

        # Save results
        results_path = output_dir / "optimal_thresholds.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved optimal thresholds to: {results_path}")

    elif args.model:
        # Single model analysis
        print(f"\nLoading model from {args.model}...")
        model = joblib.load(args.model)

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print(f"\nFinding optimal threshold (optimizing for {args.metric})...")
        optimal_threshold, metrics = find_optimal_threshold(
            y_test, y_pred_proba, metric=args.metric, min_precision=args.min_precision
        )

        print(f"\n{'=' * 60}")
        print(f"Optimal Threshold: {metrics['threshold']:.3f}")
        print(f"{'=' * 60}")
        print(f"Precision:          {metrics['precision']:.3f}")
        print(f"Recall:             {metrics['recall']:.3f}")
        print(f"F1 Score:           {metrics['f1_score']:.3f}")
        print(f"Predicted churn rate: {metrics['predicted_churn_rate']:.1%}")
        print(f"{'=' * 60}\n")

        # Compare with default threshold (0.5)
        y_pred_default = (y_pred_proba >= 0.5).astype(int)
        default_precision = precision_score(y_test, y_pred_default, zero_division=0)
        default_recall = recall_score(y_test, y_pred_default, zero_division=0)
        default_f1 = f1_score(y_test, y_pred_default, zero_division=0)

        print("Comparison with default threshold (0.5):")
        print(f"  Precision: {default_precision:.3f} → {metrics['precision']:.3f}")
        print(f"  Recall:    {default_recall:.3f} → {metrics['recall']:.3f}")
        print(f"  F1 Score:  {default_f1:.3f} → {metrics['f1_score']:.3f}")

        # Plot threshold analysis
        model_name = Path(args.model).stem
        plot_path = output_dir / f"{model_name}_threshold_analysis.png"
        plot_threshold_analysis(y_test, y_pred_proba, optimal_threshold, plot_path)

    else:
        print("Error: Specify either --model or --compare-all")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
