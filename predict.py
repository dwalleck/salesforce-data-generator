"""Make predictions using trained churn prediction models.

Load saved models and generate predictions on new customer data.
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path: Path | str):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    return model


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for prediction by dropping non-predictive columns.

    Args:
        df: Input DataFrame with features

    Returns:
        DataFrame with features ready for prediction
    """
    # Columns to drop (same as in model_trainer.py)
    drop_cols = ["account_id", "month", "last_touchbase_date", "enabled_channels"]

    # Also drop 'churned' if it exists (for when predicting on labeled data)
    if "churned" in df.columns:
        drop_cols.append("churned")

    # Drop columns that exist
    cols_to_drop = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=cols_to_drop)

    return X


def assign_risk_tier(probability: float) -> str:
    """
    Assign risk tier based on churn probability.

    Tiers:
        - High Risk (70%+): Personal outreach from account manager
        - Medium Risk (40-70%): Automated email campaign, product tips
        - Low Risk (20-40%): Monitor, no immediate action
        - Very Safe (<20%): Upsell opportunity

    Args:
        probability: Churn probability (0-1)

    Returns:
        Risk tier label
    """
    if probability >= 0.7:
        return "High Risk"
    elif probability >= 0.4:
        return "Medium Risk"
    elif probability >= 0.2:
        return "Low Risk"
    else:
        return "Very Safe"


def make_predictions(
    model,
    X: pd.DataFrame,
    threshold: float | None = None,
    include_risk_tiers: bool = True,
) -> pd.DataFrame:
    """
    Make predictions with risk scores and actionable tiers.

    Args:
        model: Trained model
        X: Features DataFrame
        threshold: Custom prediction threshold (default: 0.5)
        include_risk_tiers: Whether to include risk tier labels

    Returns:
        DataFrame with churn_risk_score, prediction, and risk_tier
    """
    # Get probability scores
    probabilities = model.predict_proba(X)
    churn_probability = probabilities[:, 1]

    # Binary predictions using threshold
    if threshold is not None:
        predictions = (churn_probability >= threshold).astype(int)
    else:
        predictions = model.predict(X)

    # Build result DataFrame
    result_df = pd.DataFrame({
        "churn_risk_score": churn_probability,
        "prediction": predictions,
        "safe_probability": probabilities[:, 0],
    })

    # Add risk tiers
    if include_risk_tiers:
        result_df["risk_tier"] = result_df["churn_risk_score"].apply(assign_risk_tier)

    return result_df


def main() -> int:
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(
        description="Make churn predictions using trained models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Make predictions with a specific model
  python predict.py --model models_3month/xgboost.joblib --input data/new_customers.csv

  # Save predictions to file
  python predict.py --model models_3month/xgboost.joblib --input data/customers.csv --output predictions.csv

  # Include original data in output
  python predict.py --model models_3month/xgboost.joblib --input data/customers.csv --include-original
        """,
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.joblib file)"
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file with customer features"
    )

    parser.add_argument(
        "--output", type=str, help="Path to save predictions CSV (optional, prints to stdout if not specified)"
    )

    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include original data columns in output",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help="Custom prediction threshold (default: 0.5). Use lower values (e.g., 0.2) to catch more at-risk customers.",
    )

    args = parser.parse_args()

    # Load model
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Load input data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        return 1

    print(f"  Loaded {len(df)} records")

    # Keep original data for output if requested
    if args.include_original:
        original_df = df.copy()

    # Prepare features
    print("Preparing features...")
    X = prepare_features_for_prediction(df)

    # Make predictions
    print("Making predictions...")
    if args.threshold:
        print(f"  Using custom threshold: {args.threshold}")
    predictions_df = make_predictions(model, X, threshold=args.threshold, include_risk_tiers=True)

    # Combine with original data if requested
    if args.include_original:
        output_df = pd.concat([original_df.reset_index(drop=True), predictions_df], axis=1)
    else:
        # Include account_id and month if they exist
        id_cols = []
        if "account_id" in df.columns:
            id_cols.append("account_id")
        if "month" in df.columns:
            id_cols.append("month")

        if id_cols:
            output_df = pd.concat([df[id_cols].reset_index(drop=True), predictions_df], axis=1)
        else:
            output_df = predictions_df

    # Save or display results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_df.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to: {output_path}")

        # Print summary
        print(f"\n{'=' * 60}")
        print("PREDICTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total records: {len(output_df)}")

        # Risk tier breakdown
        if "risk_tier" in output_df.columns:
            print(f"\nRisk Tier Distribution:")
            tier_counts = output_df["risk_tier"].value_counts()
            for tier in ["High Risk", "Medium Risk", "Low Risk", "Very Safe"]:
                count = tier_counts.get(tier, 0)
                pct = count / len(output_df) * 100
                print(f"  {tier:15} {count:6} ({pct:5.1f}%)")

        # Risk score statistics
        if "churn_risk_score" in output_df.columns:
            print(f"\nRisk Score Statistics:")
            print(f"  Average:  {output_df['churn_risk_score'].mean():.1%}")
            print(f"  Median:   {output_df['churn_risk_score'].median():.1%}")
            print(f"  Max:      {output_df['churn_risk_score'].max():.1%}")
            print(f"  Min:      {output_df['churn_risk_score'].min():.1%}")

        # Binary predictions (if threshold was used)
        if "prediction" in output_df.columns:
            churn_count = (output_df["prediction"] == 1).sum()
            churn_rate = churn_count / len(output_df)
            print(f"\nBinary Predictions:")
            print(f"  Predicted churns:     {churn_count:6} ({churn_rate:.1%})")
            print(f"  Predicted non-churns: {len(output_df) - churn_count:6} ({1-churn_rate:.1%})")

        print(f"{'=' * 60}\n")

    else:
        print("\nPredictions:")
        print(output_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
