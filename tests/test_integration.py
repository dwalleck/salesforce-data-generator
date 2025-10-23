"""Integration tests for end-to-end pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from feature_engineer import generate_features, load_data, save_data
from synthetic_generator import generate_customer_data


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test the complete data generation and feature engineering pipeline."""

    def test_basic_pipeline(self):
        """Test basic end-to-end pipeline: generate → feature engineer."""
        # Generate synthetic data
        df = generate_customer_data(num_customers=50, num_months=12, churn_rate=0.2, random_state=42)

        assert len(df) > 0, "No data generated"
        original_row_count = len(df)

        # Apply feature engineering
        df_features = generate_features(df)

        # Verify output
        assert len(df_features) == original_row_count, "Row count changed"
        assert len(df_features.columns) > len(df.columns), "No features added"

        # Verify no NaNs
        assert df_features.isna().sum().sum() == 0, "NaN values found in output"

    def test_pipeline_with_file_io(self):
        """Test end-to-end pipeline with file I/O."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw_data.csv"
            features_path = Path(tmpdir) / "features.csv"

            # Generate and save synthetic data
            df = generate_customer_data(num_customers=30, num_months=12, churn_rate=0.3, random_state=123)
            save_data(df, str(raw_path), output_format="csv")

            # Load and feature engineer
            df_loaded = load_data(str(raw_path))
            df_features = generate_features(df_loaded)

            # Save features
            save_data(df_features, str(features_path), output_format="csv")

            # Verify files exist
            assert raw_path.exists(), "Raw data file not created"
            assert features_path.exists(), "Features file not created"

            # Load features back and verify
            df_final = load_data(str(features_path))
            assert len(df_final) == len(df), "Row count mismatch after I/O"

    def test_reproducibility(self):
        """Test that same random_state produces identical results."""
        # Run 1
        df1 = generate_customer_data(num_customers=20, num_months=12, random_state=42)
        df1_features = generate_features(df1)

        # Run 2
        df2 = generate_customer_data(num_customers=20, num_months=12, random_state=42)
        df2_features = generate_features(df2)

        # Compare
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
        pd.testing.assert_frame_equal(df1_features, df2_features, check_dtype=False)

    def test_pipeline_with_edge_cases(self):
        """Test pipeline handles edge cases correctly."""
        # Zero churn rate
        df_no_churn = generate_customer_data(num_customers=10, num_months=6, churn_rate=0.0, random_state=42)
        df_no_churn_features = generate_features(df_no_churn)
        assert df_no_churn_features["churned"].sum() == 0

        # Full churn rate
        df_all_churn = generate_customer_data(num_customers=10, num_months=6, churn_rate=1.0, random_state=42)
        df_all_churn_features = generate_features(df_all_churn)
        assert df_all_churn_features[df_all_churn_features["churned"] == 1]["account_id"].nunique() == 10

        # Minimum months
        df_min_months = generate_customer_data(num_customers=10, num_months=3, churn_rate=0.3, random_state=42)
        df_min_months_features = generate_features(df_min_months)
        assert len(df_min_months_features) > 0


@pytest.mark.integration
class TestDataSchemaValidation:
    """Validate the schema of generated and feature-engineered data."""

    def test_synthetic_data_schema(self):
        """Verify synthetic data has expected schema."""
        df = generate_customer_data(num_customers=10, num_months=6, random_state=42)

        expected_columns = [
            "account_id",
            "month",
            "current_month_transactions",
            "current_month_revenue",
            "total_tickets",
            "escalated_tickets",
            "late_payments",
            "enabled_channels",
            "self_service_percentage",
            "last_touchbase_date",
            "average_resolution_time_hours",
            "churned",
        ]

        assert list(df.columns) == expected_columns, (
            f"Schema mismatch. Expected: {expected_columns}, Got: {list(df.columns)}"
        )

    def test_feature_engineered_schema(self):
        """Verify feature-engineered data contains expected features."""
        df = generate_customer_data(num_customers=10, num_months=12, random_state=42)
        df_features = generate_features(df)

        # Original columns should still be present
        original_cols = df.columns.tolist()
        for col in original_cols:
            assert col in df_features.columns, f"Original column {col} missing after feature engineering"

        # Key feature categories should be present
        assert any("_90d_" in col for col in df_features.columns), "No 90d rolling features found"
        assert any("_180d_" in col for col in df_features.columns), "No 180d rolling features found"
        assert any("percent_change" in col for col in df_features.columns), "No percent_change features found"
        assert any("acceleration" in col for col in df_features.columns), "No acceleration features found"
        assert "number_of_channels" in df_features.columns, "number_of_channels feature missing"
        assert "account_age_months" in df_features.columns, "account_age_months feature missing"

    def test_data_types(self):
        """Verify data types are appropriate."""
        df = generate_customer_data(num_customers=10, num_months=6, random_state=42)
        df_features = generate_features(df)

        # Numeric columns should be numeric
        numeric_cols = [
            "current_month_transactions",
            "current_month_revenue",
            "total_tickets",
            "escalated_tickets",
            "late_payments",
            "self_service_percentage",
            "average_resolution_time_hours",
        ]

        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df_features[col]), f"{col} is not numeric type"

        # Binary columns should be 0 or 1
        binary_cols = ["churned", "has_web", "has_card", "has_agent", "has_ivr", "has_apple_pay"]
        for col in binary_cols:
            assert df_features[col].isin([0, 1]).all(), f"{col} contains non-binary values"


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Test performance with larger datasets."""

    def test_large_dataset_generation(self):
        """Test generation with larger dataset (1000 customers, 24 months)."""
        df = generate_customer_data(num_customers=1000, num_months=24, churn_rate=0.15, random_state=42)

        assert len(df) > 10000, "Expected > 10k records"
        assert df["account_id"].nunique() == 1000, "Expected 1000 unique customers"

    def test_feature_engineering_performance(self):
        """Test feature engineering on larger dataset."""
        df = generate_customer_data(num_customers=500, num_months=24, churn_rate=0.15, random_state=42)

        df_features = generate_features(df)

        assert len(df_features) == len(df), "Row count changed"
        assert df_features.isna().sum().sum() == 0, "NaN values found"
