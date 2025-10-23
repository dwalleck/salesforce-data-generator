"""Analyze what drives customer churn using feature importance and segmentation.

Provides actionable insights by identifying key churn drivers and customer segments.
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from model_trainer import prepare_features, time_based_split
from predict import make_predictions


def analyze_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract and analyze feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with features and importance scores
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model {type(model).__name__} does not have feature_importances_")

    importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )

    return importance_df.head(top_n)


def categorize_feature(feature_name: str) -> str:
    """
    Categorize features by type for better interpretation.

    Args:
        feature_name: Feature column name

    Returns:
        Category label
    """
    if "revenue" in feature_name:
        return "Revenue/Spending"
    elif "transaction" in feature_name:
        return "Transaction Volume"
    elif "ticket" in feature_name or "escalated" in feature_name:
        return "Support Quality"
    elif "channel" in feature_name:
        return "Channel Usage"
    elif "payment" in feature_name:
        return "Payment Behavior"
    elif "self_service" in feature_name:
        return "Self-Service"
    elif "resolution" in feature_name:
        return "Support Speed"
    elif "contact" in feature_name or "touchbase" in feature_name:
        return "Engagement"
    elif "age" in feature_name:
        return "Customer Tenure"
    else:
        return "Other"


def plot_feature_importance_by_category(
    importance_df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> None:
    """
    Plot feature importance grouped by category.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        output_path: Path to save plot (optional)
    """
    # Add categories
    importance_df = importance_df.copy()
    importance_df["category"] = importance_df["feature"].apply(categorize_feature)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top features
    top_features = importance_df.head(15)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

    ax1.barh(range(len(top_features)), top_features["importance"].values, color=colors)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features["feature"].values)
    ax1.set_xlabel("Importance Score")
    ax1.set_title("Top 15 Features Driving Churn")
    ax1.invert_yaxis()

    # Category aggregation
    category_importance = importance_df.groupby("category")["importance"].sum().sort_values(ascending=False)

    colors2 = plt.cm.plasma(np.linspace(0, 1, len(category_importance)))
    ax2.barh(range(len(category_importance)), category_importance.values, color=colors2)
    ax2.set_yticks(range(len(category_importance)))
    ax2.set_yticklabels(category_importance.index)
    ax2.set_xlabel("Total Importance")
    ax2.set_title("Feature Importance by Category")
    ax2.invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved feature importance analysis: {output_path}")
    else:
        plt.show()

    plt.close()


def segment_by_churn_drivers(
    df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    top_features: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Segment high-risk customers by what's driving their churn risk.

    Args:
        df: Original data with features
        predictions_df: Predictions with risk scores
        top_features: List of top feature names

    Returns:
        Dictionary mapping segment names to customer DataFrames
    """
    # Combine data
    combined_df = pd.concat([df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)

    # Focus on high-risk customers
    high_risk = combined_df[combined_df["risk_tier"].isin(["High Risk", "Medium Risk"])].copy()

    if len(high_risk) == 0:
        print("  No high-risk customers found for segmentation")
        return {}

    segments = {}

    # Segment 1: Support issues (high tickets, long resolution times)
    if any("ticket" in f or "resolution" in f for f in top_features):
        support_issues = high_risk[
            (high_risk.get("total_tickets", 0) > high_risk.get("total_tickets", 0).median())
            | (
                high_risk.get("average_resolution_time_hours", 0)
                > high_risk.get("average_resolution_time_hours", 0).median()
            )
        ]
        if len(support_issues) > 0:
            segments["Support Issues"] = support_issues

    # Segment 2: Declining engagement (low transactions, fewer channels)
    if any("transaction" in f or "channel" in f for f in top_features):
        declining_engagement = high_risk[
            (
                high_risk.get("current_month_transactions", 0)
                < high_risk.get("current_month_transactions", 0).quantile(0.33)
            )
            | (high_risk.get("number_of_channels", 0) <= 2)
        ]
        if len(declining_engagement) > 0:
            segments["Declining Engagement"] = declining_engagement

    # Segment 3: Payment issues (late payments)
    if any("payment" in f for f in top_features):
        payment_issues = high_risk[high_risk.get("late_payments", 0) > 0]
        if len(payment_issues) > 0:
            segments["Payment Issues"] = payment_issues

    # Segment 4: Revenue decline
    if any("revenue" in f and "trend" in f for f in top_features):
        revenue_decline = high_risk[
            high_risk.get("current_month_revenue_trend_90d", 1.0) < 0.85  # 15% below 90-day average
        ]
        if len(revenue_decline) > 0:
            segments["Revenue Decline"] = revenue_decline

    return segments


def print_segment_insights(segments: dict[str, pd.DataFrame]) -> None:
    """
    Print actionable insights for each segment.

    Args:
        segments: Dictionary mapping segment names to DataFrames
    """
    print(f"\n{'=' * 60}")
    print("CUSTOMER SEGMENTATION & RECOMMENDED ACTIONS")
    print(f"{'=' * 60}\n")

    segment_actions = {
        "Support Issues": "Assign dedicated support rep, proactive check-ins, prioritize tickets",
        "Declining Engagement": "Onboarding campaign, product tutorials, feature adoption emails",
        "Payment Issues": "Flexible payment plans, financial hardship outreach, billing support",
        "Revenue Decline": "Show ROI evidence, success stories, upsell value-add features",
    }

    for segment_name, segment_df in segments.items():
        print(f"{segment_name}:")
        print(f"  Count: {len(segment_df)} customers")

        if "account_id" in segment_df.columns:
            avg_risk = segment_df["churn_risk_score"].mean()
            print(f"  Average churn risk: {avg_risk:.1%}")

        action = segment_actions.get(segment_name, "Review and take appropriate action")
        print(f"  Recommended Action:")
        print(f"    → {action}")
        print()


def main() -> int:
    """Main entry point for churn driver analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze what drives customer churn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze XGBoost model
  python analyze_churn_drivers.py --input data/features.csv --model models_3month/xgboost.joblib

  # Analyze and create customer segments
  python analyze_churn_drivers.py --input data/features.csv \\
      --model models_3month/xgboost.joblib --segment

  # Export high-risk segments for action
  python analyze_churn_drivers.py --input data/features.csv \\
      --model models_3month/xgboost.joblib --segment --export-segments
        """,
    )

    parser.add_argument("--input", type=str, required=True, help="Path to feature-engineered CSV file")

    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.joblib)")

    parser.add_argument("--output-dir", type=str, default="results_3month", help="Directory to save results (default: results_3month/)")

    parser.add_argument("--segment", action="store_true", help="Perform customer segmentation")

    parser.add_argument("--export-segments", action="store_true", help="Export segments to CSV files")

    parser.add_argument("--top-n", type=int, default=20, help="Number of top features to analyze (default: 20)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Load model
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)

    model_name = Path(args.model).stem

    # Split data (use test set for analysis)
    print("Splitting data...")
    _, _, test_df = time_based_split(df, train_months=9, val_months=2, test_months=1)

    X_test, y_test = prepare_features(test_df)
    print(f"  Test set: {len(X_test)} samples")

    # Feature importance analysis
    print(f"\nAnalyzing feature importance...")
    importance_df = analyze_feature_importance(model, X_test.columns.tolist(), top_n=args.top_n)

    print(f"\nTop {min(10, args.top_n)} Features Driving Churn:")
    print(f"{'=' * 60}")
    for idx, row in importance_df.head(10).iterrows():
        category = categorize_feature(row["feature"])
        print(f"  {row['feature']:45} {row['importance']:.4f}  [{category}]")

    # Plot feature importance
    plot_path = output_dir / f"{model_name}_churn_drivers.png"
    plot_feature_importance_by_category(importance_df, plot_path)

    # Customer segmentation
    if args.segment:
        print(f"\nPerforming customer segmentation...")

        # Get predictions with risk scores
        predictions_df = make_predictions(model, X_test, threshold=None, include_risk_tiers=True)

        # Segment by churn drivers
        segments = segment_by_churn_drivers(test_df, predictions_df, importance_df["feature"].tolist())

        if segments:
            print_segment_insights(segments)

            # Export segments
            if args.export_segments:
                segments_dir = output_dir / "segments"
                segments_dir.mkdir(exist_ok=True)

                for segment_name, segment_df in segments.items():
                    filename = segment_name.lower().replace(" ", "_") + ".csv"
                    segment_path = segments_dir / filename

                    # Select key columns for export
                    export_cols = ["account_id", "churn_risk_score", "risk_tier"]
                    available_cols = [c for c in export_cols if c in segment_df.columns]

                    segment_df[available_cols].to_csv(segment_path, index=False)
                    print(f"  Exported {segment_name}: {segment_path}")

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
