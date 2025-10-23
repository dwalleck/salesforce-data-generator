"""Unit tests for feature_engineer.py."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from feature_engineer import (
    _add_change_features,
    _add_rolling_features,
    generate_features,
    load_data,
    save_data,
)
from synthetic_generator import generate_customer_data


@pytest.fixture
def sample_data():
    """Create sample customer data for testing."""
    return generate_customer_data(num_customers=10, num_months=12, churn_rate=0.2, random_state=42)


class TestRollingFeatures:
    """Test rolling feature calculations."""

    def test_rolling_features_created(self, sample_data):
        """Verify rolling features are created for specified windows."""
        df = sample_data.copy()
        windows = {"90d": (3, 1), "180d": (6, 3)}

        df = _add_rolling_features(df, "current_month_revenue", windows)

        # Check that all expected columns exist
        expected_cols = [
            "current_month_revenue_90d_sum",
            "current_month_revenue_90d_avg",
            "current_month_revenue_90d_max",
            "current_month_revenue_90d_min",
            "current_month_revenue_trend_90d",
            "current_month_revenue_volatility_90d",
            "current_month_revenue_180d_sum",
            "current_month_revenue_180d_avg",
            "current_month_revenue_180d_max",
            "current_month_revenue_180d_min",
            "current_month_revenue_trend_180d",
            "current_month_revenue_volatility_180d",
        ]

        for col in expected_cols:
            assert col in df.columns, f"Missing rolling feature: {col}"

    def test_rolling_sum_accuracy(self, sample_data):
        """Verify rolling sum is calculated correctly."""
        df = sample_data.copy()
        df = df.sort_values(["account_id", "month"])

        windows = {"90d": (3, 1)}
        df = _add_rolling_features(df, "current_month_transactions", windows)

        # For a specific customer, verify the 3-month rolling sum
        customer_df = df[df["account_id"] == df["account_id"].iloc[0]].reset_index(drop=True)

        if len(customer_df) >= 3:
            # 4th month should have sum of months 2, 3, 4 (index 1, 2, 3)
            expected_sum = customer_df.loc[1:3, "current_month_transactions"].sum()
            actual_sum = customer_df.loc[3, "current_month_transactions_90d_sum"]
            assert abs(actual_sum - expected_sum) < 0.01, (
                f"Rolling sum mismatch: expected {expected_sum}, got {actual_sum}"
            )


class TestChangeFeatures:
    """Test change feature calculations."""

    def test_change_features_created(self, sample_data):
        """Verify percent change and acceleration features are created."""
        df = sample_data.copy()
        df = _add_change_features(df, "current_month_revenue")

        assert "current_month_revenue_percent_change" in df.columns
        assert "current_month_revenue_acceleration" in df.columns

    def test_percent_change_accuracy(self, sample_data):
        """Verify percent change is calculated correctly."""
        df = sample_data.copy()
        df = df.sort_values(["account_id", "month"])
        df = _add_change_features(df, "current_month_transactions")

        # For a specific customer, manually verify percent change
        customer_df = df[df["account_id"] == df["account_id"].iloc[0]].reset_index(drop=True)

        if len(customer_df) >= 2:
            prev_value = customer_df.loc[0, "current_month_transactions"]
            curr_value = customer_df.loc[1, "current_month_transactions"]
            expected_pct_change = (curr_value - prev_value) / prev_value if prev_value > 0 else 0

            actual_pct_change = customer_df.loc[1, "current_month_transactions_percent_change"]

            # Account for potential NaN if prev_value is 0
            if pd.isna(actual_pct_change) and prev_value == 0:
                pass  # This is expected
            else:
                assert abs(actual_pct_change - expected_pct_change) < 0.01, (
                    f"Percent change mismatch: expected {expected_pct_change}, got {actual_pct_change}"
                )


class TestGenerateFeatures:
    """Test the main generate_features function."""

    def test_basic_feature_generation(self, sample_data):
        """Verify features are generated successfully."""
        original_cols = len(sample_data.columns)
        df = generate_features(sample_data)

        assert len(df.columns) > original_cols, "No features were added"
        assert len(df) == len(sample_data), "Row count changed during feature generation"

    def test_channel_features_created(self, sample_data):
        """Verify binary channel features are created."""
        df = generate_features(sample_data)

        channel_features = ["has_web", "has_card", "has_agent", "has_ivr", "has_apple_pay"]

        for feature in channel_features:
            assert feature in df.columns, f"Missing channel feature: {feature}"
            assert df[feature].isin([0, 1]).all(), f"{feature} contains non-binary values"

    def test_number_of_channels_calculated(self, sample_data):
        """Verify number_of_channels is sum of binary channel flags."""
        df = generate_features(sample_data)

        channel_cols = ["has_web", "has_card", "has_agent", "has_ivr", "has_apple_pay"]
        expected_count = df[channel_cols].sum(axis=1)
        actual_count = df["number_of_channels"]

        pd.testing.assert_series_equal(actual_count, expected_count, check_names=False)

    def test_missing_columns_raises_error(self):
        """Missing required columns should raise ValueError."""
        incomplete_df = pd.DataFrame(
            {
                "account_id": ["ACC-0001"],
                "month": ["2024-01-01"],
                "current_month_revenue": [1000.0],
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_features(incomplete_df)

    def test_empty_dataframe_raises_error(self):
        """Empty DataFrame should raise ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            generate_features(empty_df)


class TestFileIO:
    """Test load_data and save_data functions."""

    def test_load_save_csv(self, sample_data):
        """Test loading and saving CSV files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            # Save as CSV
            save_data(sample_data, temp_path, output_format="csv")

            # Load back
            loaded_df = load_data(temp_path)

            # Compare (some columns may have type differences)
            assert len(loaded_df) == len(sample_data)
            assert list(loaded_df.columns) == list(sample_data.columns)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_save_json(self, sample_data):
        """Test loading and saving JSON files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save as JSON
            save_data(sample_data, temp_path, output_format="json")

            # Load back
            loaded_df = load_data(temp_path)

            assert len(loaded_df) == len(sample_data)
            assert list(loaded_df.columns) == list(sample_data.columns)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_save_parquet(self, sample_data):
        """Test loading and saving Parquet files."""
        # Skip if parquet support not available
        try:
            import pyarrow
        except ImportError:
            pytest.skip("Parquet support (pyarrow) not installed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            # Save as Parquet
            save_data(sample_data, temp_path, output_format="parquet")

            # Load back
            loaded_df = load_data(temp_path)

            assert len(loaded_df) == len(sample_data)
            assert list(loaded_df.columns) == list(sample_data.columns)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_nonexistent_file_raises_error(self):
        """Loading non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/to/file.csv")

    def test_save_unsupported_format_raises_error(self, sample_data):
        """Saving with unsupported format should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            save_data(sample_data, "output.txt", output_format="txt")


class TestDataIntegrity:
    """Property-based tests using Hypothesis."""

    @given(
        num_customers=st.integers(min_value=5, max_value=50),
        num_months=st.integers(min_value=6, max_value=12),
        churn_rate=st.floats(min_value=0.0, max_value=1.0),
        random_state=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=10, deadline=10000)
    def test_no_nans_in_output(self, num_customers, num_months, churn_rate, random_state):
        """Verify fillna(0) eliminates all NaNs from feature-engineered data."""
        df = generate_customer_data(
            num_customers=num_customers, num_months=num_months, churn_rate=churn_rate, random_state=random_state
        )

        df_features = generate_features(df)

        # Check for NaNs in all columns
        nan_counts = df_features.isna().sum()
        assert nan_counts.sum() == 0, f"Found NaN values in columns: {nan_counts[nan_counts > 0].to_dict()}"

    @given(
        num_customers=st.integers(min_value=5, max_value=50),
        num_months=st.integers(min_value=6, max_value=12),
        random_state=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=10, deadline=10000)
    def test_account_age_increases_monotonically(self, num_customers, num_months, random_state):
        """Account age should increase monotonically for each customer."""
        df = generate_customer_data(num_customers=num_customers, num_months=num_months, random_state=random_state)

        df_features = generate_features(df)

        for account_id in df_features["account_id"].unique():
            customer_df = df_features[df_features["account_id"] == account_id]
            ages = customer_df["account_age_months"].values
            # Check that ages are monotonically increasing
            assert (np.diff(ages) >= 0).all(), f"Account age not monotonic for {account_id}: {ages}"
