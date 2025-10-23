"""Explain individual customer churn predictions using feature values and importance.

Shows WHY specific customers are predicted to churn by analyzing
which of their features are most abnormal compared to safe customers.
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_trainer import prepare_features, time_based_split


def explain_prediction_simple(
    model,
    X: pd.DataFrame,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    customer_idx: int,
    customer_id: str | None = None,
) -> dict:
    """
    Explain a customer's prediction by comparing their features to safe vs churned customers.

    Args:
        model: Trained model with feature_importances_
        X: Features DataFrame (for prediction)
        X_full: Full features DataFrame (for comparison)
        y_full: Full labels (for comparison)
        customer_idx: Index of customer to explain
        customer_id: Optional customer ID for display

    Returns:
        Dictionary with explanation details
    """
    # Get customer's features
    customer_features = X.iloc[customer_idx]

    # Get prediction
    pred_proba = model.predict_proba(X.iloc[customer_idx:customer_idx + 1])[0, 1]
    prediction = model.predict(X.iloc[customer_idx:customer_idx + 1])[0]

    # Get feature importance from model
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    else:
        feature_importance = np.ones(len(X.columns)) / len(X.columns)

    # Calculate statistics for safe vs churned customers
    safe_customers = X_full[y_full == 0]
    churned_customers = X_full[y_full == 1] if (y_full == 1).any() else None

    # Analyze each feature
    feature_analysis = []

    for i, feature_name in enumerate(X.columns):
        customer_value = customer_features.iloc[i]
        importance = feature_importance[i]

        # Compare to safe customers
        safe_mean = safe_customers[feature_name].mean()
        safe_std = safe_customers[feature_name].std()

        # Z-score: how many standard deviations from safe customer average
        if safe_std > 0:
            z_score = (customer_value - safe_mean) / safe_std
        else:
            z_score = 0

        # Compare to churned customers if available
        if churned_customers is not None and len(churned_customers) > 0:
            churned_mean = churned_customers[feature_name].mean()
            closer_to_churned = abs(customer_value - churned_mean) < abs(customer_value - safe_mean)
        else:
            churned_mean = None
            closer_to_churned = False

        feature_analysis.append({
            'feature': feature_name,
            'value': float(customer_value),
            'safe_avg': float(safe_mean),
            'churned_avg': float(churned_mean) if churned_mean is not None else None,
            'z_score': float(z_score),
            'importance': float(importance),
            'risk_score': abs(z_score) * importance,  # Combined abnormality + importance
            'closer_to_churned': closer_to_churned,
        })

    # Sort by risk score
    feature_analysis.sort(key=lambda x: x['risk_score'], reverse=True)

    # Create explanation
    explanation = {
        'customer_id': customer_id or f"Customer_{customer_idx}",
        'churn_probability': float(pred_proba),
        'prediction': int(prediction),
        'top_risk_factors': feature_analysis[:10],
    }

    # Print explanation
    print(f"\n{'=' * 75}")
    print(f"EXPLANATION FOR: {explanation['customer_id']}")
    print(f"{'=' * 75}")
    print(f"Churn Probability: {pred_proba:.1%}")
    print(f"Prediction: {'⚠️  WILL CHURN' if prediction == 1 else '✅ SAFE'}")
    print(f"\nTop 10 Risk Factors (Why This Customer Is At Risk):")
    print(f"{'─' * 75}\n")

    for i, factor in enumerate(feature_analysis[:10], 1):
        # Determine if value is concerning
        if factor['z_score'] > 2:
            concern = "⚠️  VERY HIGH"
        elif factor['z_score'] > 1:
            concern = "↑ High"
        elif factor['z_score'] < -2:
            concern = "⚠️  VERY LOW"
        elif factor['z_score'] < -1:
            concern = "↓ Low"
        else:
            concern = "~ Normal"

        print(f"{i:2}. {factor['feature'][:50]:<50}")
        print(f"    Customer's value: {factor['value']:>10.2f}  [{concern}]")
        print(f"    Safe customers avg: {factor['safe_avg']:>10.2f}")

        if factor['churned_avg'] is not None:
            print(f"    Churned customers avg: {factor['churned_avg']:>10.2f}")
            if factor['closer_to_churned']:
                print(f"    → This customer looks like churned customers ⚠️")

        if abs(factor['z_score']) > 2:
            if factor['z_score'] > 0:
                print(f"    → {abs(factor['z_score']):.1f}σ ABOVE safe customer average!")
            else:
                print(f"    → {abs(factor['z_score']):.1f}σ BELOW safe customer average!")

        print()

    return explanation


def main() -> int:
    """Main entry point for simple prediction explanations."""
    parser = argparse.ArgumentParser(
        description="Explain individual churn predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain top 5 highest-risk customers
  python explain_predictions_simple.py --input data/features.csv --model models_3month/xgboost.joblib --top-n 5

  # Explain a specific customer
  python explain_predictions_simple.py --input data/features.csv --model models_3month/xgboost.joblib --customer ACC-0123
        """,
    )

    parser.add_argument("--input", type=str, required=True, help="Path to feature-engineered CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--customer", type=str, help="Specific customer ID to explain")
    parser.add_argument("--top-n", type=int, default=3, help="Number of highest-risk customers to explain (default: 3)")

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

    if args.customer:
        # Explain specific customer
        if 'account_id' not in test_df.columns:
            print("Error: account_id column not found in data")
            return 1

        # Find customer in test set
        test_df_reset = test_df.reset_index(drop=True)
        customer_mask = test_df_reset['account_id'] == args.customer

        if not customer_mask.any():
            print(f"Error: Customer {args.customer} not found in test set")
            available = test_df_reset['account_id'].unique()[:10]
            print(f"Available customers (first 10): {', '.join(available)}")
            return 1

        customer_idx = customer_mask.idxmax()

        explain_prediction_simple(
            model,
            X_test,
            X_test,
            y_test,
            customer_idx=customer_idx,
            customer_id=args.customer,
        )

    else:
        # Explain top N highest-risk customers
        print(f"\nExplaining top {args.top_n} highest-risk customers...\n")

        # Find top N
        top_indices = np.argsort(y_pred_proba)[-args.top_n:][::-1]

        test_df_reset = test_df.reset_index(drop=True)
        account_ids = test_df_reset['account_id'] if 'account_id' in test_df_reset.columns else pd.Series(range(len(X_test)))

        for rank, idx in enumerate(top_indices, 1):
            print(f"\n{'#' * 75}")
            print(f"HIGH RISK CUSTOMER #{rank} (Risk: {y_pred_proba[idx]:.1%})")
            print(f"{'#' * 75}")

            explain_prediction_simple(
                model,
                X_test,
                X_test,
                y_test,
                customer_idx=idx,
                customer_id=str(account_ids.iloc[idx]),
            )

        print(f"\n{'=' * 75}")
        print(f"Explained {args.top_n} high-risk customers")
        print(f"{'=' * 75}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
