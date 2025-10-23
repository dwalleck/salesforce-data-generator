# Customer Churn Prediction - Synthetic Data Generator

A Python toolkit for generating realistic synthetic customer data with temporal patterns and churn signals, designed for training machine learning models for customer churn prediction.

## Overview

This project provides two main tools:

1. **Synthetic Data Generator** (`synthetic_generator.py`) - Creates realistic customer lifecycle data with five distinct churn patterns
2. **Feature Engineering Pipeline** (`feature_engineer.py`) - Transforms raw data into 112 ML-ready temporal features

The generated data includes realistic churn signals that appear 3 months before churn, making it ideal for developing and testing churn prediction models.

## Features

### Data Generator
- **Five Churn Cohorts**: early_churn, renewal_churn, gradual_disengagement, service_issues, price_sensitive
- **12 Customer Metrics**: transactions, revenue, support tickets, channels, payments, and more
- **Temporal Patterns**: Progressive degradation in the 3 months preceding churn
- **Configurable**: 30+ parameters via `GeneratorConfig` dataclass
- **Multiple Formats**: Export to CSV, JSON, or Parquet

### Feature Engineering
- **112 Features**: Comprehensive temporal aggregations and transformations
- **Multi-Window Analysis**: 90-day and 180-day rolling windows
- **Trend Detection**: Percent change, acceleration, volatility metrics
- **Channel Analytics**: Channel adoption, abandonment, and trend features
- **Production-Ready**: Input validation, NaN handling, type hints

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/salesforce_data_generator.git
cd salesforce_data_generator

# Install dependencies
uv sync
```

## Quick Start

### Complete Workflow (Generate → Feature Engineer → Train → Predict)

```bash
# Run the complete pipeline
./examples/train_and_evaluate.sh
```

### Generate Synthetic Data

```bash
# Generate 1000 customers with 12 months of history
uv run python synthetic_generator.py -o customer_data.csv

# Generate larger dataset with higher churn rate
uv run python synthetic_generator.py -n 10000 -m 24 -c 0.25 -o data.csv
```

### Apply Feature Engineering

```bash
# Generate features from CSV
uv run python feature_engineer.py customer_data.csv

# Specify output file
uv run python feature_engineer.py data.csv -o enriched_data.csv

# Convert to JSON format
uv run python feature_engineer.py data.csv --format json
```

### Train ML Models

```bash
# Train all models with default settings
uv run python model_trainer.py --input data/features.csv

# With hyperparameter tuning (recommended)
uv run python model_trainer.py --input data/features.csv --tune

# Train specific models only
uv run python model_trainer.py --input data/features.csv --models xgb rf

# Custom time splits (train=8mo, val=2mo, test=2mo)
uv run python model_trainer.py --input data/features.csv \
    --train-months 8 --val-months 2 --test-months 2
```

### Make Predictions

```bash
# Predict with trained model (3-month prediction window)
uv run python predict.py \
    --model models_3month/xgboost.joblib \
    --input data/new_customers.csv \
    --output predictions.csv

# Include original data in output
uv run python predict.py \
    --model models_3month/xgboost.joblib \
    --input data/customers.csv \
    --include-original \
    --output full_predictions.csv
```

## Project Structure

```
salesforce_data_generator/
│
├── synthetic_generator.py              # Generate synthetic customer churn data
├── feature_engineer.py                 # Transform raw data into ML features
├── model_trainer_3month.py             # Train 3-month churn prediction model (primary)
├── model_trainer_1month.py             # Train 1-month model (archived, for reference)
├── model_evaluator.py                  # Model evaluation and visualization utilities
├── predict.py                          # Make predictions on new customer data
├── analyze_churn_drivers.py            # Global feature importance & customer segmentation
├── explain_predictions_simple.py       # Individual customer risk explanations
├── explain_predictions.py              # SHAP-based explanations (alternative)
├── optimize_threshold.py               # Find optimal prediction thresholds
│
├── README.md                           # Project overview and usage guide
├── 3MONTH_PREDICTION_GUIDE.md          # Guide for 3-month prediction workflow
├── ACTIONABLE_INSIGHTS_GUIDE.md        # Risk scores, segmentation & actions
├── INDIVIDUAL_EXPLANATIONS_GUIDE.md    # Feature-level customer explanations
│
├── pyproject.toml                      # Project dependencies (uv)
├── uv.lock                             # Dependency lock file
├── pytest.ini                          # Test configuration
│
├── tests/                              # Test suite (69 tests total)
│   ├── test_synthetic_generator.py     # Data generation tests (29 tests)
│   ├── test_feature_engineer.py        # Feature engineering tests (16 tests)
│   ├── test_model_trainer.py           # ML training tests (14 tests)
│   └── test_integration.py             # End-to-end tests (10 tests)
│
├── data/                               # Generated datasets (created by scripts)
│   ├── raw_customer_data.csv           # Raw customer data
│   ├── features.csv                    # Feature-engineered data
│   └── predictions*.csv                # Prediction outputs
│
├── models_3month/                      # Trained models (created by model_trainer_3month.py)
│   └── xgboost.joblib                  # XGBoost model (3-month predictions)
│
├── results_3month/                     # Model evaluation (created by model_trainer_3month.py)
│   ├── metrics.json                    # Model performance metrics
│   ├── roc_curves_comparison.png       # ROC curve
│   ├── pr_curves_comparison.png        # Precision-Recall curve
│   ├── confusion_matrices/
│   │   └── xgboost_confusion_matrix.png
│   └── feature_importance/
│       └── xgboost_feature_importance.png
│
└── examples/
    └── train_and_evaluate.sh           # Complete workflow example
```

**Note**: Directories marked with `(created by scripts)` are generated when you run the corresponding commands and won't exist in a fresh clone.

**Key Scripts by Use Case:**

| Use Case | Script | Description |
|----------|--------|-------------|
| **Generate Data** | `synthetic_generator.py` | Create synthetic customer churn dataset |
| **Prepare Features** | `feature_engineer.py` | Engineer 107 ML features from raw data |
| **Train Model** | `model_trainer_3month.py` | Train XGBoost for 3-month predictions |
| **Make Predictions** | `predict.py` | Score customers with risk probabilities |
| **Understand Why** | `analyze_churn_drivers.py` | Global feature importance |
| **Segment Customers** | `analyze_churn_drivers.py --segment` | Group by churn drivers |
| **Explain Individual** | `explain_predictions_simple.py` | Why specific customer at risk |
| **Optimize Threshold** | `optimize_threshold.py` | Find best decision boundary |
| **Run Tests** | `pytest` | Validate all functionality (69 tests) |

## Data Schema

### Raw Data (from synthetic_generator.py)

| Column | Type | Description |
|--------|------|-------------|
| `account_id` | str | Unique customer identifier (ACC-XXXX) |
| `month` | date | Record month (YYYY-MM-DD) |
| `current_month_transactions` | int | Transaction count for the month |
| `current_month_revenue` | float | Revenue for the month (rounded to 2 decimals) |
| `total_tickets` | int | Support tickets opened |
| `escalated_tickets` | int | Escalated support tickets |
| `late_payments` | int | Number of late payments |
| `enabled_channels` | str | Pipe-separated channel list (web\|card\|agent\|ivr\|apple_pay) |
| `self_service_percentage` | float | Self-service adoption rate (0-100) |
| `last_touchbase_date` | datetime | Most recent customer contact (ISO format) |
| `average_resolution_time_hours` | float | Average ticket resolution time |
| `churned` | int | Binary flag (1 = churned this month) |

### Engineered Features (from feature_engineer.py)

**112 total features** organized in categories:

1. **Channel Features** (10): `has_web`, `has_card`, `has_agent`, `has_ivr`, `has_apple_pay`, `number_of_channels`, `channel_trend`, `channels_dropped`, etc.

2. **Customer Metadata** (2): `days_since_last_contact`, `account_age_months`

3. **Revenue Features** (24): 90d/180d sums, averages, max, min, trends, volatility, percent change, acceleration

4. **Transaction Features** (24): Same temporal features as revenue

5. **Support Features** (24): Temporal features for `late_payments` and `escalated_tickets`

6. **Service Quality Features** (20): Temporal features for `self_service_percentage` and `average_resolution_time_hours`

All features use consistent time windows:
- **90d (3 months)**: Recent trends, immediate pre-churn signals
- **180d (6 months)**: Long-term patterns, seasonal changes

## ML Training & Evaluation

This project includes a complete machine learning pipeline for training and evaluating churn prediction models.

### Components

1. **`model_trainer.py`** - Train multiple models with hyperparameter tuning
   - Logistic Regression (baseline)
   - Random Forest (ensemble method)
   - XGBoost (gradient boosting)
   - Time-based train/validation/test split (prevents data leakage)
   - SMOTE for class imbalance handling
   - TimeSeriesSplit cross-validation for hyperparameter tuning

2. **`model_evaluator.py`** - Comprehensive evaluation utilities
   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
   - Visualizations: ROC curves, PR curves, confusion matrices, feature importance
   - Model comparison plots
   - JSON metrics export

3. **`predict.py`** - Inference on new data
   - Load trained models
   - Make predictions with probability scores
   - Export results to CSV

### Key Features

**Time-Based Splitting**: Prevents data leakage by splitting chronologically (e.g., train on months 1-9, validate on 10-11, test on 12). This mimics real-world production scenarios where you train on historical data and predict future churn.

**Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance training data, improving model performance on the minority class (churned customers).

**Hyperparameter Tuning**: Grid search with TimeSeriesSplit cross-validation to find optimal parameters while respecting temporal ordering.

**Comprehensive Evaluation**: Focus on recall and precision (not just accuracy), which are more important for churn prediction where catching churners is critical.

### Model Training Output

After training with `model_trainer_3month.py` (3-month prediction window), the following artifacts are generated:

```
models_3month/
└── xgboost.joblib                 # Trained XGBoost model (3-month predictions)

results_3month/
├── metrics.json                   # Performance metrics
├── roc_curves_comparison.png      # ROC curve
├── pr_curves_comparison.png       # Precision-Recall curve
├── confusion_matrices/
│   └── xgboost_confusion_matrix.png
└── feature_importance/
    └── xgboost_feature_importance.png
```

**Note**: We focus on 3-month predictions (90-day action window) using XGBoost, which provides the best balance of performance and actionability. For reference, the 1-month trainer (`model_trainer_1month.py`) is available but not recommended for production use.

### Example: Train and Evaluate

```python
# Generate data
from synthetic_generator import generate_customer_data
from feature_engineer import generate_features

df = generate_customer_data(num_customers=1000, num_months=12, churn_rate=0.15)
df_features = generate_features(df)
df_features.to_csv('data/features.csv', index=False)

# Train models (via CLI)
# uv run python model_trainer.py --input data/features.csv --tune --models all

# Or train programmatically
from model_trainer import time_based_split, prepare_features, train_model, get_model_configs
from model_evaluator import evaluate_model, print_evaluation_summary

# Time-based split
train_df, val_df, test_df = time_based_split(df_features, train_months=9, val_months=2, test_months=1)

X_train, y_train = prepare_features(train_df)
X_test, y_test = prepare_features(test_df)

# Train XGBoost
configs = get_model_configs(tune=False)
model, params = train_model(X_train, y_train, "xgboost", *configs["xgboost"])

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name="XGBoost")
print_evaluation_summary(metrics)
```

## Churn Cohorts

The generator creates five distinct churn patterns with realistic leading indicators:

### 1. Early Churn (Months 1-3)
**Poor onboarding experience**
- High support ticket volume (2x baseline)
- Low self-service adoption (50% of baseline)
- Channel abandonment (50% removal probability)

### 2. Renewal Churn (Months 11-13)
**Contract renewal issues**
- Churns at contract renewal period
- No specific degradation pattern (decision-based)

### 3. Gradual Disengagement (Month 6+)
**Slow decline in engagement**
- 15% monthly decline in transactions and revenue
- 20% monthly decline in support tickets
- 10 additional days between contacts per month

### 4. Service Issues (Month 6+)
**Poor service quality**
- Support ticket spike (3x baseline)
- Increased resolution time (1.5x baseline)
- 20 additional days between contacts per month

### 5. Price Sensitive (Month 6+)
**Financial constraints**
- Revenue decline to 70% of baseline
- 4x higher late payment probability (40% vs 10%)
- Stable transaction volume

### 6. Stable (Never Churns)
**Healthy customers**
- Normal variance in all metrics
- No degradation patterns

## Configuration

The generator supports extensive configuration through the `GeneratorConfig` dataclass:

```python
from synthetic_generator import GeneratorConfig

config = GeneratorConfig(
    # Baseline characteristics
    baseline_transactions_min=50,
    baseline_transactions_max=300,
    baseline_revenue_multiplier_min=50.0,
    baseline_revenue_multiplier_max=500.0,

    # Monthly variance (±10%)
    transaction_variance_min=0.9,
    transaction_variance_max=1.1,

    # Channel probabilities
    channel_web_prob=0.80,
    channel_card_prob=0.70,
    channel_agent_prob=0.60,
    channel_ivr_prob=0.40,
    channel_apple_pay_prob=0.20,

    # Churn signal parameters (30+ configurable parameters)
    gradual_disengagement_decline_rate=0.15,
    service_issues_ticket_multiplier=3,
    price_sensitive_revenue_min_factor=0.7,
    # ... see GeneratorConfig in synthetic_generator.py for all options
)

df = generate_customer_data(num_customers=1000, config=config)
```

## Example Workflow

Complete end-to-end example:

```python
from synthetic_generator import generate_customer_data
from feature_engineer import generate_features
import pandas as pd

# Step 1: Generate synthetic data
print("Generating synthetic customer data...")
df = generate_customer_data(
    num_customers=5000,
    num_months=18,
    churn_rate=0.20,
    random_state=42
)

print(f"Generated {len(df)} records for {df['account_id'].nunique()} customers")
print(f"Churn rate: {df['churned'].sum() / df['account_id'].nunique():.1%}")

# Step 2: Apply feature engineering
print("\nApplying feature engineering...")
df_features = generate_features(df)

print(f"Created {len(df_features.columns)} features")
print(f"No NaN values: {df_features.isna().sum().sum() == 0}")

# Step 3: Split into train/test
from sklearn.model_selection import train_test_split

# Get the last record for each customer (decision point)
df_final = df_features.groupby('account_id').last().reset_index()

# Separate features and target
X = df_final.drop(['account_id', 'month', 'churned'], axis=1)
y = df_final['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Train churn rate: {y_train.mean():.1%}")
print(f"Test churn rate: {y_test.mean():.1%}")

# Step 4: Train a model (example with XGBoost)
import xgboost as xgb

model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
```

## Feature Engineering Details

### Rolling Window Features

For each metric, the following features are generated for 90d and 180d windows:

```python
# For column 'current_month_revenue':
current_month_revenue_90d_sum         # Sum over last 3 months
current_month_revenue_90d_avg         # Average over last 3 months
current_month_revenue_90d_max         # Maximum in last 3 months
current_month_revenue_90d_min         # Minimum in last 3 months
current_month_revenue_trend_90d       # Current / 90d average
current_month_revenue_volatility_90d  # Standard deviation over 90 days

# And the same for 180d (6 months)
current_month_revenue_180d_sum
current_month_revenue_180d_avg
# ... etc
```

### Change Features

Month-over-month changes and acceleration:

```python
current_month_revenue_percent_change  # (current - previous) / previous
current_month_revenue_acceleration    # Change in percent_change
```

### Comparison Features

```python
current_month_revenue_30d                  # Current month value
current_month_revenue_recent_vs_historical # Current / (prev_month_1 + prev_month_2)
```

### NaN Handling

All NaN values (which occur in derived features for early months) are filled with 0, which is safe because:
- Division by zero is prevented via offset parameters in all calculations
- 0 is semantically appropriate for "no change yet" scenarios

## Development

### Code Quality

This project follows Python best practices:
- [x] Type hints on all functions
- [x] Comprehensive docstrings (PEP 257)
- [x] Input validation with clear error messages
- [x] Line length limit: 120 characters
- [x] Modular design with single-responsibility functions
- [x] Extensive inline comments for complex algorithms

### Testing

```bash
# Run basic import tests
uv run python -c "from synthetic_generator import generate_customer_data; print('Generator OK')"
uv run python -c "from feature_engineer import generate_features; print('Feature engineer OK')"

# Run end-to-end test
uv run python -c "
from synthetic_generator import generate_customer_data
from feature_engineer import generate_features

df = generate_customer_data(num_customers=10, num_months=6)
df_features = generate_features(df)
assert df_features.isna().sum().sum() == 0, 'NaN values found'
print(f'Generated {len(df_features.columns)} features with no NaNs')
"
```

## Performance Considerations

The feature engineering pipeline may show performance warnings for large datasets due to iterative column additions. For production use with very large datasets (>100K customers), consider:

1. Using `pd.concat(axis=1)` to add features in batches
2. Processing data in chunks by customer cohorts
3. Pre-allocating DataFrame with all column names

For typical use cases (<50K customers), current performance is acceptable.

## Contributing

Contributions are welcome! Please ensure:
- All functions have type hints and docstrings
- Code follows PEP 8 (120 char line limit)
- New features include explanatory comments
- Changes maintain backward compatibility

## License

[MIT License](LICENSE)

## Acknowledgments

- Inspired by real-world customer churn patterns in SaaS businesses
- Built with Python, Pandas, and NumPy

## Contact

For questions or feedback, please open an issue on GitHub.
