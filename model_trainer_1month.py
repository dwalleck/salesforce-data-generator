"""Train churn prediction models with time-based splitting and hyperparameter tuning.

Implements multiple ML models with cross-validation, class imbalance handling,
and comprehensive evaluation.
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from model_evaluator import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    print_evaluation_summary,
    save_metrics,
)

warnings.filterwarnings("ignore", category=UserWarning)


def time_based_split(
    df: pd.DataFrame, train_months: int, val_months: int, test_months: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by time (month) to prevent data leakage.

    Args:
        df: DataFrame with 'month' column
        train_months: Number of months for training
        val_months: Number of months for validation
        test_months: Number of months for testing

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        ValueError: If split months exceed available data
    """
    df = df.sort_values("month").copy()
    unique_months = sorted(df["month"].unique())

    total_months = train_months + val_months + test_months
    if len(unique_months) < total_months:
        raise ValueError(
            f"Insufficient data: need {total_months} months, have {len(unique_months)}. "
            f"Adjust --train-months, --val-months, --test-months parameters."
        )

    train_end_idx = train_months
    val_end_idx = train_months + val_months

    train_months_list = unique_months[:train_end_idx]
    val_months_list = unique_months[train_end_idx:val_end_idx]
    test_months_list = unique_months[val_end_idx : val_end_idx + test_months]

    train_df = df[df["month"].isin(train_months_list)]
    val_df = df[df["month"].isin(val_months_list)]
    test_df = df[df["month"].isin(test_months_list)]

    return train_df, val_df, test_df


def prepare_features(df: pd.DataFrame, drop_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for modeling by dropping non-predictive columns.

    Args:
        df: Input DataFrame with features and target
        drop_cols: Additional columns to drop

    Returns:
        Tuple of (X, y) where X is features, y is target
    """
    # Default columns to drop (non-predictive or leakage risk)
    default_drop = ["account_id", "month", "last_touchbase_date", "churned", "enabled_channels"]

    if drop_cols:
        default_drop.extend(drop_cols)

    # Remove duplicates
    drop_cols_final = list(set(default_drop))

    # Drop columns that exist
    cols_to_drop = [col for col in drop_cols_final if col in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df["churned"]

    # Replace inf/-inf with NaN, then fill NaN with 0
    # This handles edge cases from division operations
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y


def handle_class_imbalance(X_train: pd.DataFrame, y_train: pd.Series, method: str = "smote") -> tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance in training data.

    Args:
        X_train: Training features
        y_train: Training labels
        method: Imbalance handling method ('smote', 'none')

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if method == "smote":
        print(f"  Original class distribution: {y_train.value_counts().to_dict()}")

        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train) - 1))
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"  After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled, name=y_train.name)

        return X_resampled, y_resampled
    else:
        return X_train, y_train


def get_model_configs(tune: bool = False) -> dict:
    """
    Get model configurations with hyperparameter grids.

    Args:
        tune: Whether to include hyperparameter tuning grids

    Returns:
        Dictionary mapping model names to (model_class, params) tuples
    """
    configs = {}

    if tune:
        # Hyperparameter grids for tuning
        configs["logistic_regression"] = (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"], "solver": ["lbfgs"]},
        )

        configs["random_forest"] = (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        )

        configs["xgboost"] = (
            XGBClassifier(random_state=42, eval_metric="logloss"),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0],
            },
        )
    else:
        # Default parameters (no tuning)
        configs["logistic_regression"] = (LogisticRegression(max_iter=1000, random_state=42), {})

        configs["random_forest"] = (RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42), {})

        configs["xgboost"] = (
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss"),
            {},
        )

    return configs


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    model_class,
    param_grid: dict,
    tune: bool = False,
    cv_splits: int = 3,
) -> tuple:
    """
    Train a single model with optional hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
        model_class: Model class instance
        param_grid: Hyperparameter grid for tuning
        tune: Whether to perform hyperparameter tuning
        cv_splits: Number of cross-validation splits

    Returns:
        Tuple of (trained_model, best_params)
    """
    print(f"\nTraining {model_name}...")

    if tune and param_grid:
        print(f"  Performing hyperparameter tuning with {cv_splits}-fold CV...")

        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        grid_search = GridSearchCV(
            model_class, param_grid, cv=tscv, scoring="f1", n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train, y_train)

        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV F1 score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_
    else:
        model_class.fit(X_train, y_train)
        return model_class, {}


def main() -> int:
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train churn prediction models with time-based splitting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default parameters
  python model_trainer.py --input data/features.csv

  # With hyperparameter tuning
  python model_trainer.py --input data/features.csv --tune

  # Train specific models only
  python model_trainer.py --input data/features.csv --models lr xgb

  # Custom time splits
  python model_trainer.py --input data/features.csv --train-months 8 --val-months 2 --test-months 2
        """,
    )

    parser.add_argument("--input", type=str, required=True, help="Path to feature-engineered CSV file")

    parser.add_argument(
        "--output-dir", type=str, default="models", help="Directory to save trained models (default: models/)"
    )

    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to save results (default: results/)"
    )

    parser.add_argument("--train-months", type=int, default=9, help="Number of months for training (default: 9)")

    parser.add_argument("--val-months", type=int, default=2, help="Number of months for validation (default: 2)")

    parser.add_argument("--test-months", type=int, default=1, help="Number of months for testing (default: 1)")

    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")

    parser.add_argument("--cv-splits", type=int, default=3, help="Number of CV splits for tuning (default: 3)")

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lr", "rf", "xgb", "all"],
        default=["all"],
        help="Models to train: lr (logistic regression), rf (random forest), xgb (xgboost), all",
    )

    parser.add_argument(
        "--imbalance-method",
        choices=["smote", "none"],
        default="smote",
        help="Method to handle class imbalance (default: smote)",
    )

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrices").mkdir(exist_ok=True)
    (results_dir / "feature_importance").mkdir(exist_ok=True)

    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        return 1

    print(f"  Loaded {len(df)} records")
    print(f"  Unique customers: {df['account_id'].nunique()}")

    # Time-based split
    print(f"\nSplitting data: train={args.train_months}, val={args.val_months}, test={args.test_months} months...")
    try:
        train_df, val_df, test_df = time_based_split(df, args.train_months, args.val_months, args.test_months)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"  Train: {len(train_df)} records")
    print(f"  Validation: {len(val_df)} records")
    print(f"  Test: {len(test_df)} records")

    # Prepare features
    print("\nPreparing features...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    print(f"  Feature count: {X_train.shape[1]}")
    print(f"  Train churn rate: {y_train.mean():.1%}")
    print(f"  Val churn rate: {y_val.mean():.1%}")
    print(f"  Test churn rate: {y_test.mean():.1%}")

    # Handle class imbalance
    print(f"\nHandling class imbalance using method: {args.imbalance_method}")
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train, method=args.imbalance_method)

    # Determine which models to train
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

        # Evaluate on test set
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

        # Plot confusion matrix
        cm_path = results_dir / "confusion_matrices" / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)

        # Plot feature importance (for tree-based models)
        if hasattr(trained_model, "feature_importances_"):
            fi_path = results_dir / "feature_importance" / f"{model_name}_feature_importance.png"
            plot_feature_importance(
                X_train.columns.tolist(), trained_model.feature_importances_, model_name, top_n=20, output_path=fi_path
            )

    # Comparison plots
    print("\nGenerating comparison plots...")
    plot_roc_curve(y_test, y_pred_proba_dict, output_path=results_dir / "roc_curves_comparison.png")
    plot_precision_recall_curve(y_test, y_pred_proba_dict, output_path=results_dir / "pr_curves_comparison.png")

    # Save all metrics
    save_metrics(all_metrics, output_path=results_dir / "metrics.json")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Models saved to: {output_dir}")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
