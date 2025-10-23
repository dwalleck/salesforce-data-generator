#!/bin/bash
# Example workflow for generating data, training models, and making predictions

set -e  # Exit on error

echo "=== Salesforce Churn Prediction - Complete Workflow ==="
echo

# Create data directory
mkdir -p data

# Step 1: Generate synthetic customer data
echo "[1/5] Generating synthetic customer data..."
uv run python synthetic_generator.py \
    --num-customers 1000 \
    --num-months 12 \
    --churn-rate 0.15 \
    --random-seed 42 \
    --output data/raw_customer_data.csv

echo "✓ Generated data/raw_customer_data.csv"
echo

# Step 2: Feature engineering
echo "[2/5] Performing feature engineering..."
uv run python feature_engineer.py \
    data/raw_customer_data.csv \
    --output data/features.csv

echo "✓ Generated data/features.csv"
echo

# Step 3: Train models with hyperparameter tuning
echo "[3/5] Training models (this may take a few minutes)..."
uv run python model_trainer.py \
    --input data/features.csv \
    --output-dir models \
    --results-dir results \
    --train-months 9 \
    --val-months 2 \
    --test-months 1 \
    --tune \
    --models all \
    --imbalance-method smote

echo "✓ Models trained and saved to models/"
echo "✓ Results saved to results/"
echo

# Step 4: Make predictions on test data
echo "[4/5] Making predictions with best model..."
uv run python predict.py \
    --model models/xgboost.joblib \
    --input data/features.csv \
    --output data/predictions.csv

echo "✓ Predictions saved to data/predictions.csv"
echo

# Step 5: Display results summary
echo "[5/5] Results Summary"
echo "===================="
echo

if [ -f results/metrics.json ]; then
    echo "Model Performance Metrics:"
    cat results/metrics.json | head -20
    echo "..."
    echo
fi

echo "Generated Files:"
echo "  Data:"
echo "    - data/raw_customer_data.csv      (raw synthetic data)"
echo "    - data/features.csv                (feature-engineered data)"
echo "    - data/predictions.csv             (model predictions)"
echo
echo "  Models:"
echo "    - models/logistic_regression.joblib"
echo "    - models/random_forest.joblib"
echo "    - models/xgboost.joblib"
echo
echo "  Results:"
echo "    - results/metrics.json             (performance metrics)"
echo "    - results/roc_curves_comparison.png"
echo "    - results/pr_curves_comparison.png"
echo "    - results/confusion_matrices/*"
echo "    - results/feature_importance/*"
echo

echo "=== Workflow Complete! ==="
echo
echo "Next steps:"
echo "  - View visualizations in results/"
echo "  - Review metrics in results/metrics.json"
echo "  - Use predict.py with new data for inference"
