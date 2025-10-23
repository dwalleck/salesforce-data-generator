"""Explain individual customer churn predictions using SHAP.

Shows WHY specific customers are predicted to churn by breaking down
the contribution of each feature to their risk score.
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from model_trainer import prepare_features, time_based_split


def explain_prediction(
    model,
    X: pd.DataFrame,
    customer_idx: int,
    customer_id: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Explain a single customer's churn prediction using SHAP.

    Args:
        model: Trained model
        X: Features DataFrame
        customer_idx: Index of customer to explain
        customer_id: Optional customer ID for display
        output_dir: Directory to save plots

    Returns:
        Dictionary with explanation details
    """
    # Create SHAP explainer
    print(f"Creating SHAP explainer (this may take a moment)...")
    explainer = shap.TreeExplainer(model)

    # Get SHAP values for this customer
    customer_features = X.iloc[customer_idx:customer_idx + 1]
    shap_values = explainer.shap_values(customer_features)

    # For binary classification, get values for class 1 (churn)
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1][0]
    else:
        shap_values_churn = shap_values[0]

    # Get prediction
    pred_proba = model.predict_proba(customer_features)[0, 1]
    prediction = model.predict(customer_features)[0]

    # Get base value (expected value)
    if isinstance(explainer.expected_value, list):
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    # Create explanation dictionary
    explanation = {
        "customer_id": customer_id or f"Customer_{customer_idx}",
        "churn_probability": float(pred_proba),
        "prediction": int(prediction),
        "base_probability": float(base_value),
        "features": {},
    }

    # Get top contributing features
    feature_impacts = []
    for i, (feature_name, shap_value) in enumerate(zip(X.columns, shap_values_churn)):
        feature_value = customer_features.iloc[0, i]
        feature_impacts.append({
            "feature": feature_name,
            "value": float(feature_value),
            "impact": float(shap_value),
            "impact_pct": float(shap_value) * 100,
        })

    # Sort by absolute impact
    feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
    explanation["top_drivers"] = feature_impacts[:10]

    # Print explanation
    print(f"\n{'=' * 70}")
    print(f"EXPLANATION FOR: {explanation['customer_id']}")
    print(f"{'=' * 70}")
    print(f"Churn Probability: {pred_proba:.1%}")
    print(f"Prediction: {'WILL CHURN' if prediction == 1 else 'SAFE'}")
    print(f"Base Probability: {base_value:.1%} (average customer)")
    print(f"\nTop 10 Factors Driving This Prediction:")
    print(f"{'─' * 70}")

    for i, driver in enumerate(feature_impacts[:10], 1):
        impact_dir = "↑" if driver["impact"] > 0 else "↓"
        impact_text = "increases" if driver["impact"] > 0 else "decreases"

        print(f"{i:2}. {driver['feature'][:45]:<45}")
        print(f"    Value: {driver['value']:>10.2f}")
        print(f"    Impact: {impact_dir} {impact_text} risk by {abs(driver['impact_pct']):.1f} percentage points")
        print()

    # Generate visualizations
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Waterfall plot
        fig = plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_churn,
                base_values=base_value,
                data=customer_features.values[0],
                feature_names=X.columns.tolist(),
            ),
            show=False,
        )
        waterfall_path = output_dir / f"{explanation['customer_id']}_waterfall.png"
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved waterfall plot: {waterfall_path}")

        # Force plot (saved as HTML)
        force_plot = shap.force_plot(
            base_value,
            shap_values_churn,
            customer_features,
            feature_names=X.columns.tolist(),
            matplotlib=False,
        )
        force_path = output_dir / f"{explanation['customer_id']}_force_plot.html"
        shap.save_html(str(force_path), force_plot)
        print(f"  Saved force plot: {force_path}")

    return explanation


def explain_high_risk_customers(
    model,
    X: pd.DataFrame,
    y_pred_proba: np.ndarray,
    account_ids: pd.Series,
    top_n: int = 5,
    output_dir: Path | None = None,
) -> list[dict]:
    """
    Explain predictions for top N highest-risk customers.

    Args:
        model: Trained model
        X: Features DataFrame
        y_pred_proba: Predicted probabilities
        account_ids: Customer account IDs
        top_n: Number of high-risk customers to explain
        output_dir: Directory to save plots

    Returns:
        List of explanation dictionaries
    """
    # Find top N highest risk customers
    top_indices = np.argsort(y_pred_proba)[-top_n:][::-1]

    explanations = []
    for rank, idx in enumerate(top_indices, 1):
        print(f"\n{'#' * 70}")
        print(f"HIGH RISK CUSTOMER #{rank} (Risk: {y_pred_proba[idx]:.1%})")
        print(f"{'#' * 70}")

        explanation = explain_prediction(
            model,
            X,
            customer_idx=idx,
            customer_id=str(account_ids.iloc[idx]),
            output_dir=output_dir / f"top_{rank}" if output_dir else None,
        )

        explanations.append(explanation)

    return explanations


def main() -> int:
    """Main entry point for prediction explanations."""
    parser = argparse.ArgumentParser(
        description="Explain individual churn predictions using SHAP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain top 5 highest-risk customers
  python explain_predictions.py --input data/features.csv --model models_3month/xgboost.joblib --top-n 5

  # Explain a specific customer
  python explain_predictions.py --input data/features.csv --model models_3month/xgboost.joblib --customer ACC-0123

  # Save visualizations
  python explain_predictions.py --input data/features.csv --model models_3month/xgboost.joblib --top-n 3 --output results_3month/explanations
        """,
    )

    parser.add_argument("--input", type=str, required=True, help="Path to feature-engineered CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--customer", type=str, help="Specific customer ID to explain (e.g., ACC-0123)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of highest-risk customers to explain (default: 5)")
    parser.add_argument("--output", type=str, help="Directory to save explanation plots")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Load model
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)

    # Split data (use test set)
    print("Splitting data...")
    _, _, test_df = time_based_split(df, train_months=9, val_months=2, test_months=1)

    X_test, y_test = prepare_features(test_df)
    print(f"  Test set: {len(X_test)} customers")

    # Get predictions
    print("Making predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Prepare output directory
    output_dir = Path(args.output) if args.output else None

    if args.customer:
        # Explain specific customer
        if 'account_id' not in test_df.columns:
            print("Error: account_id column not found in data")
            return 1

        # Find customer
        customer_mask = test_df['account_id'] == args.customer
        if not customer_mask.any():
            print(f"Error: Customer {args.customer} not found in test set")
            available = test_df['account_id'].unique()[:10]
            print(f"Available customers (first 10): {', '.join(available)}")
            return 1

        customer_idx = test_df[customer_mask].index[0]
        test_idx = test_df.index.get_loc(customer_idx)

        explain_prediction(
            model,
            X_test,
            customer_idx=test_idx,
            customer_id=args.customer,
            output_dir=output_dir,
        )

    else:
        # Explain top N highest-risk customers
        print(f"\nExplaining top {args.top_n} highest-risk customers...")

        account_ids = test_df['account_id'] if 'account_id' in test_df.columns else pd.Series(range(len(X_test)))

        explanations = explain_high_risk_customers(
            model,
            X_test,
            y_pred_proba,
            account_ids,
            top_n=args.top_n,
            output_dir=output_dir,
        )

        # Summary
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"Explained {len(explanations)} high-risk customers")
        if output_dir:
            print(f"Visualizations saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
