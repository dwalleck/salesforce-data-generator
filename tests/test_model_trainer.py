"""Tests for model training pipeline."""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from feature_engineer import generate_features
from model_evaluator import evaluate_model, plot_confusion_matrix
from model_trainer import (
    get_model_configs,
    handle_class_imbalance,
    prepare_features,
    time_based_split,
    train_model,
)
from synthetic_generator import generate_customer_data


@pytest.fixture
def sample_churn_data():
    """Generate sample churn data for testing."""
    df = generate_customer_data(num_customers=100, num_months=12, churn_rate=0.2, random_state=42)

    # Apply feature engineering
    df = generate_features(df)

    return df


class TestTimeBasedSplit:
    """Test time-based data splitting."""

    def test_basic_split(self, sample_churn_data):
        """Test basic time-based split."""
        train_df, val_df, test_df = time_based_split(sample_churn_data, train_months=8, val_months=2, test_months=2)

        # Check sizes
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        # Check no overlap
        train_months = set(train_df["month"].unique())
        val_months = set(val_df["month"].unique())
        test_months = set(test_df["month"].unique())

        assert len(train_months & val_months) == 0, "Train and val months overlap"
        assert len(train_months & test_months) == 0, "Train and test months overlap"
        assert len(val_months & test_months) == 0, "Val and test months overlap"

    def test_split_preserves_all_data(self, sample_churn_data):
        """Test that split preserves all data."""
        train_df, val_df, test_df = time_based_split(sample_churn_data, train_months=9, val_months=2, test_months=1)

        total_records = len(train_df) + len(val_df) + len(test_df)
        assert total_records == len(sample_churn_data), "Data lost during split"

    def test_insufficient_months_raises_error(self, sample_churn_data):
        """Test that insufficient months raises ValueError."""
        with pytest.raises(ValueError, match="Insufficient data"):
            time_based_split(sample_churn_data, train_months=20, val_months=5, test_months=5)

    def test_temporal_ordering(self, sample_churn_data):
        """Test that train < val < test in time."""
        train_df, val_df, test_df = time_based_split(sample_churn_data, train_months=8, val_months=2, test_months=2)

        max_train_month = train_df["month"].max()
        min_val_month = val_df["month"].min()
        max_val_month = val_df["month"].max()
        min_test_month = test_df["month"].min()

        assert max_train_month < min_val_month, "Train and val not in temporal order"
        assert max_val_month < min_test_month, "Val and test not in temporal order"


class TestPrepareFeatures:
    """Test feature preparation."""

    def test_drops_non_predictive_columns(self, sample_churn_data):
        """Test that non-predictive columns are dropped."""
        X, y = prepare_features(sample_churn_data)

        # These should be dropped
        assert "account_id" not in X.columns
        assert "month" not in X.columns
        assert "churned" not in X.columns
        assert "enabled_channels" not in X.columns

        # Target should be extracted
        assert y.name == "churned"
        assert len(X) == len(y)

    def test_preserves_feature_columns(self, sample_churn_data):
        """Test that feature columns are preserved."""
        X, y = prepare_features(sample_churn_data)

        # Some expected features should be present
        expected_features = ["current_month_revenue", "current_month_transactions", "total_tickets"]

        for feature in expected_features:
            assert feature in X.columns, f"Feature {feature} missing"


class TestHandleClassImbalance:
    """Test class imbalance handling."""

    def test_smote_balances_classes(self, sample_churn_data):
        """Test that SMOTE balances class distribution."""
        X, y = prepare_features(sample_churn_data)

        # Split data
        train_df, _, _ = time_based_split(sample_churn_data, train_months=9, val_months=2, test_months=1)

        X_train, y_train = prepare_features(train_df)

        # Original imbalance
        original_churn_ratio = y_train.sum() / len(y_train)

        # Apply SMOTE
        X_resampled, y_resampled = handle_class_imbalance(X_train, y_train, method="smote")

        # Check balance improved
        new_churn_ratio = y_resampled.sum() / len(y_resampled)

        assert new_churn_ratio > original_churn_ratio, "SMOTE did not increase minority class"
        assert 0.4 <= new_churn_ratio <= 0.6, "SMOTE did not balance classes properly"

    def test_none_method_preserves_data(self, sample_churn_data):
        """Test that 'none' method doesn't modify data."""
        train_df, _, _ = time_based_split(sample_churn_data, train_months=9, val_months=2, test_months=1)

        X_train, y_train = prepare_features(train_df)

        X_resampled, y_resampled = handle_class_imbalance(X_train, y_train, method="none")

        assert len(X_resampled) == len(X_train)
        assert len(y_resampled) == len(y_train)


class TestModelTraining:
    """Test model training functionality."""

    def test_get_model_configs_no_tune(self):
        """Test getting model configs without tuning."""
        configs = get_model_configs(tune=False)

        assert "logistic_regression" in configs
        assert "random_forest" in configs
        assert "xgboost" in configs

        # Check that param grids are empty (no tuning)
        for model_name, (model, param_grid) in configs.items():
            assert param_grid == {}, f"{model_name} has param grid when tune=False"

    def test_get_model_configs_with_tune(self):
        """Test getting model configs with tuning."""
        configs = get_model_configs(tune=True)

        # Check that param grids are not empty
        for model_name, (model, param_grid) in configs.items():
            assert len(param_grid) > 0, f"{model_name} missing param grid when tune=True"

    def test_train_single_model(self, sample_churn_data):
        """Test training a single model."""
        train_df, _, _ = time_based_split(sample_churn_data, train_months=9, val_months=2, test_months=1)

        X_train, y_train = prepare_features(train_df)

        configs = get_model_configs(tune=False)
        model_class, param_grid = configs["logistic_regression"]

        trained_model, best_params = train_model(
            X_train, y_train, "logistic_regression", model_class, param_grid, tune=False, cv_splits=3
        )

        # Check model is trained
        assert trained_model is not None
        assert hasattr(trained_model, "predict")
        assert hasattr(trained_model, "predict_proba")


class TestModelEvaluation:
    """Test model evaluation utilities."""

    def test_evaluate_model_basic(self):
        """Test basic model evaluation."""
        # Create dummy predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9, 0.7, 0.2])

        metrics = evaluate_model(y_true, y_pred, y_pred_proba, model_name="test_model")

        # Check metrics present
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "confusion_matrix" in metrics

        # Check confusion matrix structure
        cm = metrics["confusion_matrix"]
        assert "tn" in cm
        assert "fp" in cm
        assert "fn" in cm
        assert "tp" in cm

    def test_plot_confusion_matrix_saves_file(self):
        """Test that confusion matrix plot is saved."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cm.png"

            plot_confusion_matrix(y_true, y_pred, "test_model", output_path)

            assert output_path.exists(), "Confusion matrix plot not saved"


class TestModelPersistence:
    """Test model saving and loading."""

    def test_save_and_load_model(self, sample_churn_data):
        """Test saving and loading trained model."""
        train_df, _, test_df = time_based_split(sample_churn_data, train_months=9, val_months=2, test_months=1)

        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)

        # Train a simple model
        configs = get_model_configs(tune=False)
        model_class, param_grid = configs["logistic_regression"]

        trained_model, _ = train_model(X_train, y_train, "lr", model_class, param_grid, tune=False)

        # Get predictions before saving
        pred_before = trained_model.predict(X_test)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"

            joblib.dump(trained_model, model_path)

            loaded_model = joblib.load(model_path)

            # Get predictions after loading
            pred_after = loaded_model.predict(X_test)

            # Check predictions are identical
            assert np.array_equal(pred_before, pred_after), "Predictions differ after save/load"
