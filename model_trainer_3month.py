"""Train churn prediction models for 3-month prediction window.

Instead of predicting "will churn next month", this predicts
"will churn in the next 3 months", giving 90 days to take action.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Import from original trainer
from model_trainer import (
    get_model_configs,
    handle_class_imbalance,
    prepare_features,
    train_model,
)
from model_evaluator import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    print_evaluation_summary,
    save_metrics,
)


def create_3month_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create labels for 3-month churn prediction.

    For each customer-month, label = 1 if customer churns ANYTIME
    in the next 3 months, 0 otherwise.

    Args:
        df: DataFrame with 'account_id', 'month', and 'churned' columns

    Returns:
        DataFrame with new 'churned_next_3mo' column
    """
    df = df.sort_values(['account_id', 'month']).copy()

    # For each customer, check if they churn in next 3 months
    def check_future_churn(group):
        """Check if customer churns in next 3 months."""
        churned_next_3mo = []

        for idx in range(len(group)):
            # Look at next 3 rows (next 3 months)
            future_window = group.iloc[idx:idx + 4]  # Current + next 3
            # Label = 1 if ANY future churn in window
            will_churn = future_window['churned'].sum() > 0
            churned_next_3mo.append(1 if will_churn else 0)

        group['churned_next_3mo'] = churned_next_3mo
        return group

    df = df.groupby('account_id', group_keys=False).apply(check_future_churn)

    return df


def time_based_split_3month(
    df: pd.DataFrame,
    train_months: int,
    prediction_window_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data for 3-month prediction.

    Train on early months, predict for the last N months.
    Test set excludes the final 3 months (since we can't predict 3 months ahead).

    Args:
        df: DataFrame with features
        train_months: Number of months for training
        prediction_window_months: Prediction window (default: 3)

    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values('month').copy()
    unique_months = sorted(df['month'].unique())
    total_months = len(unique_months)

    if total_months < train_months + prediction_window_months:
        raise ValueError(
            f"Need at least {train_months + prediction_window_months} months, "
            f"have {total_months}"
        )

    # Train on first N months
    train_months_list = unique_months[:train_months]

    # Test on remaining months, but exclude last 3 (can't predict 3 months ahead)
    # For 12 months total, train_months=6 → test on months 7-9 (predict months 10-12)
    test_end_idx = total_months - prediction_window_months
    test_months_list = unique_months[train_months:test_end_idx]

    train_df = df[df['month'].isin(train_months_list)]
    test_df = df[df['month'].isin(test_months_list)]

    return train_df, test_df


def main() -> int:
    """Main entry point for 3-month churn prediction training."""
    parser = argparse.ArgumentParser(
        description="Train models for 3-month churn prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python model_trainer_3month.py --input data/features.csv

  # With hyperparameter tuning
  python model_trainer_3month.py --input data/features.csv --tune

  # Train specific models
  python model_trainer_3month.py --input data/features.csv --models xgb rf
        """,
    )

    parser.add_argument("--input", type=str, required=True, help="Path to feature-engineered CSV")
    parser.add_argument("--output-dir", type=str, default="models_3month", help="Directory for models")
    parser.add_argument("--results-dir", type=str, default="results_3month", help="Directory for results")
    parser.add_argument("--train-months", type=int, default=6, help="Months for training (default: 6)")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--cv-splits", type=int, default=3, help="CV splits for tuning")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lr", "rf", "xgb", "all"],
        default=["all"],
        help="Models to train",
    )
    parser.add_argument(
        "--imbalance-method",
        choices=["smote", "none"],
        default="smote",
        help="Class imbalance handling",
    )

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrices").mkdir(exist_ok=True)
    (results_dir / "feature_importance").mkdir(exist_ok=True)

    print(f"{'=' * 60}")
    print("3-MONTH CHURN PREDICTION")
    print(f"{'=' * 60}\n")

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} records")

    # Create 3-month labels
    print("\nCreating 3-month churn labels...")
    df = create_3month_labels(df)

    original_churn_rate = df['churned'].mean()
    three_month_churn_rate = df['churned_next_3mo'].mean()

    print(f"  Original (1-month) churn rate: {original_churn_rate:.1%}")
    print(f"  3-month churn rate: {three_month_churn_rate:.1%}")
    print(f"  Improvement: {three_month_churn_rate/original_churn_rate:.1f}x more churners in window")

    # Time-based split
    print(f"\nSplitting data (train on first {args.train_months} months)...")
    train_df, test_df = time_based_split_3month(df, train_months=args.train_months)

    print(f"  Train: {len(train_df)} records")
    print(f"  Test: {len(test_df)} records")

    # Prepare features with NEW target
    print("\nPreparing features...")

    # Create target variable from 3-month labels
    train_df_temp = train_df.copy()
    test_df_temp = test_df.copy()

    # Replace churned with 3-month version
    train_df_temp['churned'] = train_df_temp['churned_next_3mo']
    test_df_temp['churned'] = test_df_temp['churned_next_3mo']

    # Drop the 3-month label column to prevent data leakage!
    train_df_temp = train_df_temp.drop(columns=['churned_next_3mo'])
    test_df_temp = test_df_temp.drop(columns=['churned_next_3mo'])

    X_train, y_train = prepare_features(train_df_temp)
    X_test, y_test = prepare_features(test_df_temp)

    print(f"  Feature count: {X_train.shape[1]}")
    print(f"  Train 3-month churn rate: {y_train.mean():.1%}")
    print(f"  Test 3-month churn rate: {y_test.mean():.1%}")

    # Handle class imbalance
    print(f"\nHandling class imbalance using: {args.imbalance_method}")
    X_train_balanced, y_train_balanced = handle_class_imbalance(
        X_train, y_train, method=args.imbalance_method
    )

    # Determine models to train
    model_configs = get_model_configs(tune=args.tune)

    if "all" in args.models:
        models_to_train = list(model_configs.keys())
    else:
        model_mapping = {"lr": "logistic_regression", "rf": "random_forest", "xgb": "xgboost"}
        models_to_train = [model_mapping[m] for m in args.models]

    # Train models
    trained_models = {}
    all_metrics = {}
    y_pred_proba_dict = {}

    for model_name in models_to_train:
        model_class, param_grid = model_configs[model_name]

        # Train
        trained_model, best_params = train_model(
            X_train_balanced,
            y_train_balanced,
            model_name,
            model_class,
            param_grid,
            tune=args.tune,
            cv_splits=args.cv_splits,
        )

        trained_models[model_name] = trained_model

        # Evaluate
        print(f"\nEvaluating {model_name} on test set...")
        y_pred = trained_model.predict(X_test)
        y_pred_proba = trained_model.predict_proba(X_test)[:, 1]
        y_pred_proba_dict[model_name] = y_pred_proba

        metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name=model_name)
        all_metrics[model_name] = metrics

        print_evaluation_summary(metrics)

        # Save model
        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump(trained_model, model_path)
        print(f"  Saved model: {model_path}")

        # Confusion matrix
        cm_path = results_dir / "confusion_matrices" / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)

        # Feature importance
        if hasattr(trained_model, "feature_importances_"):
            fi_path = results_dir / "feature_importance" / f"{model_name}_feature_importance.png"
            plot_feature_importance(
                X_train.columns.tolist(),
                trained_model.feature_importances_,
                model_name,
                top_n=20,
                output_path=fi_path,
            )

    # Comparison plots
    print("\nGenerating comparison plots...")
    plot_roc_curve(y_test, y_pred_proba_dict, output_path=results_dir / "roc_curves_comparison.png")
    plot_precision_recall_curve(y_test, y_pred_proba_dict, output_path=results_dir / "pr_curves_comparison.png")

    # Save metrics
    save_metrics(all_metrics, output_path=results_dir / "metrics.json")

    print(f"\n{'=' * 60}")
    print("3-MONTH PREDICTION TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Models saved to: {output_dir}")
    print(f"Results saved to: {results_dir}")
    print(f"\nPrediction window: Next 3 months (90 days to take action)")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
