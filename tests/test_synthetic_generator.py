"""Unit tests for synthetic_generator.py."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthetic_generator import (
    Channel,
    ChurnCohort,
    GeneratorConfig,
    _generate_channels,
    generate_customer_data,
)


class TestChurnCohortMonthRanges:
    """Test ChurnCohort.get_churn_month() returns correct month ranges."""

    def test_early_churn_range(self):
        """Early churn should return months 1-3."""
        np.random.seed(42)
        for _ in range(50):
            month = ChurnCohort.EARLY_CHURN.get_churn_month(num_months=12)
            assert 1 <= month <= 3, f"Early churn month {month} not in range [1, 3]"

    def test_renewal_churn_range(self):
        """Renewal churn should return months 11-13."""
        np.random.seed(42)
        for _ in range(50):
            month = ChurnCohort.RENEWAL_CHURN.get_churn_month(num_months=14)
            assert 11 <= month <= 13, f"Renewal churn month {month} not in range [11, 13]"

    def test_gradual_disengagement_range(self):
        """Gradual disengagement should return month 6+."""
        np.random.seed(42)
        for _ in range(50):
            month = ChurnCohort.GRADUAL_DISENGAGEMENT.get_churn_month(num_months=12)
            assert 6 <= month < 12, f"Gradual disengagement month {month} not in range [6, 12)"

    def test_service_issues_range(self):
        """Service issues should return month 6+."""
        np.random.seed(42)
        for _ in range(50):
            month = ChurnCohort.SERVICE_ISSUES.get_churn_month(num_months=12)
            assert 6 <= month < 12, f"Service issues month {month} not in range [6, 12)"

    def test_price_sensitive_range(self):
        """Price sensitive should return month 6+."""
        np.random.seed(42)
        for _ in range(50):
            month = ChurnCohort.PRICE_SENSITIVE.get_churn_month(num_months=12)
            assert 6 <= month < 12, f"Price sensitive month {month} not in range [6, 12)"

    def test_stable_never_churns(self):
        """Stable cohort should return num_months (never churns)."""
        assert ChurnCohort.STABLE.get_churn_month(num_months=12) == 12
        assert ChurnCohort.STABLE.get_churn_month(num_months=24) == 24

    def test_churn_month_respects_num_months_boundary(self):
        """Churn month should never exceed num_months."""
        np.random.seed(42)
        for cohort in [
            ChurnCohort.EARLY_CHURN,
            ChurnCohort.RENEWAL_CHURN,
            ChurnCohort.GRADUAL_DISENGAGEMENT,
            ChurnCohort.SERVICE_ISSUES,
            ChurnCohort.PRICE_SENSITIVE,
        ]:
            # Test with very small num_months
            for num_months in [3, 6, 12]:
                for _ in range(20):
                    month = cohort.get_churn_month(num_months=num_months)
                    assert month < num_months, f"{cohort.value} returned month {month} >= {num_months}"


class TestInputValidation:
    """Test input validation in generate_customer_data()."""

    def test_negative_num_customers_raises_error(self):
        """Negative num_customers should raise ValueError."""
        with pytest.raises(ValueError, match="num_customers must be positive"):
            generate_customer_data(num_customers=-1)

    def test_zero_num_customers_raises_error(self):
        """Zero num_customers should raise ValueError."""
        with pytest.raises(ValueError, match="num_customers must be positive"):
            generate_customer_data(num_customers=0)

    def test_insufficient_num_months_raises_error(self):
        """num_months < 3 should raise ValueError."""
        with pytest.raises(ValueError, match="num_months must be at least 3"):
            generate_customer_data(num_customers=10, num_months=2)

    def test_churn_rate_below_zero_raises_error(self):
        """churn_rate < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="churn_rate must be between 0 and 1"):
            generate_customer_data(num_customers=10, churn_rate=-0.1)

    def test_churn_rate_above_one_raises_error(self):
        """churn_rate > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="churn_rate must be between 0 and 1"):
            generate_customer_data(num_customers=10, churn_rate=1.5)

    def test_negative_random_state_raises_error(self):
        """Negative random_state should raise ValueError."""
        with pytest.raises(ValueError, match="random_state must be non-negative"):
            generate_customer_data(num_customers=10, random_state=-1)


class TestChannelGeneration:
    """Test channel generation constraints."""

    def test_apple_pay_requires_web(self):
        """Apple Pay should force Web to be enabled."""
        np.random.seed(42)
        config = GeneratorConfig(
            channel_web_prob=0.0,  # Disable web
            channel_apple_pay_prob=1.0,  # Force apple pay
        )
        channels = _generate_channels(config)

        if Channel.APPLE_PAY in channels:
            assert Channel.WEB in channels, "Apple Pay enabled but Web is not"

    def test_at_least_one_channel_enabled(self):
        """At least one channel should always be enabled."""
        np.random.seed(42)
        config = GeneratorConfig(
            channel_web_prob=0.0,
            channel_card_prob=0.0,
            channel_agent_prob=0.0,
            channel_ivr_prob=0.0,
            channel_apple_pay_prob=0.0,
        )
        channels = _generate_channels(config)
        assert len(channels) >= 1, "No channels enabled"
        assert Channel.WEB in channels, "Default channel (Web) not enabled"


class TestGenerateCustomerDataBasic:
    """Basic smoke tests for generate_customer_data()."""

    def test_basic_generation(self):
        """Generate basic dataset with default config."""
        df = generate_customer_data(num_customers=10, num_months=12, churn_rate=0.2, random_state=42)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df["account_id"].nunique() == 10

    def test_output_columns_present(self):
        """Output should have all expected columns."""
        df = generate_customer_data(num_customers=5, num_months=6, random_state=42)

        expected_cols = [
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

        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_account_id_format(self):
        """Account IDs should follow ACC-XXXX format."""
        df = generate_customer_data(num_customers=5, num_months=6, random_state=42)

        for account_id in df["account_id"].unique():
            assert account_id.startswith("ACC-"), f"Invalid account_id format: {account_id}"
            assert len(account_id) == 8, f"Invalid account_id length: {account_id}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_churn_rate(self):
        """churn_rate=0 should produce no churned customers."""
        df = generate_customer_data(num_customers=20, num_months=12, churn_rate=0.0, random_state=42)

        assert df["churned"].sum() == 0, "Found churned customers with churn_rate=0"
        # All customers should have 12 months of data
        assert df.groupby("account_id").size().min() == 12

    def test_full_churn_rate(self):
        """churn_rate=1.0 should churn all customers."""
        df = generate_customer_data(num_customers=20, num_months=12, churn_rate=1.0, random_state=42)

        churned_customers = df[df["churned"] == 1]["account_id"].nunique()
        assert churned_customers == 20, f"Only {churned_customers}/20 customers churned"

    def test_minimum_months(self):
        """num_months=3 should work correctly."""
        df = generate_customer_data(num_customers=10, num_months=3, churn_rate=0.3, random_state=42)

        assert len(df) > 0
        assert df.groupby("account_id").size().max() <= 3

    def test_reproducibility_with_same_seed(self):
        """Same random_state should produce identical results."""
        df1 = generate_customer_data(num_customers=10, num_months=6, random_state=123)
        df2 = generate_customer_data(num_customers=10, num_months=6, random_state=123)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_results_with_different_seed(self):
        """Different random_state should produce different results."""
        df1 = generate_customer_data(num_customers=10, num_months=6, random_state=123)
        df2 = generate_customer_data(num_customers=10, num_months=6, random_state=456)

        # At least one value should differ
        assert not df1.equals(df2), "Same results with different random seeds"


class TestDataIntegrity:
    """Property-based tests using Hypothesis to verify data invariants."""

    @given(
        num_customers=st.integers(min_value=5, max_value=100),
        num_months=st.integers(min_value=3, max_value=24),
        churn_rate=st.floats(min_value=0.0, max_value=1.0),
        random_state=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_no_negative_numeric_values(self, num_customers, num_months, churn_rate, random_state):
        """All numeric columns should have non-negative values."""
        df = generate_customer_data(
            num_customers=num_customers, num_months=num_months, churn_rate=churn_rate, random_state=random_state
        )

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
            assert (df[col] >= 0).all(), f"Negative values found in {col}"

    @given(
        num_customers=st.integers(min_value=5, max_value=100),
        num_months=st.integers(min_value=3, max_value=24),
        churn_rate=st.floats(min_value=0.0, max_value=1.0),
        random_state=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_churned_flag_binary(self, num_customers, num_months, churn_rate, random_state):
        """Churned flag should only be 0 or 1."""
        df = generate_customer_data(
            num_customers=num_customers, num_months=num_months, churn_rate=churn_rate, random_state=random_state
        )

        assert df["churned"].isin([0, 1]).all(), "Churned flag contains non-binary values"

    @given(
        num_customers=st.integers(min_value=5, max_value=100),
        num_months=st.integers(min_value=3, max_value=24),
        churn_rate=st.floats(min_value=0.0, max_value=1.0),
        random_state=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_self_service_percentage_range(self, num_customers, num_months, churn_rate, random_state):
        """Self-service percentage should be between 0 and 100."""
        df = generate_customer_data(
            num_customers=num_customers, num_months=num_months, churn_rate=churn_rate, random_state=random_state
        )

        assert (df["self_service_percentage"] >= 0).all(), "Self-service % below 0"
        assert (df["self_service_percentage"] <= 100).all(), "Self-service % above 100"

    @given(
        num_customers=st.integers(min_value=5, max_value=100),
        num_months=st.integers(min_value=3, max_value=24),
        churn_rate=st.floats(min_value=0.0, max_value=1.0),
        random_state=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_escalated_tickets_not_exceed_total(self, num_customers, num_months, churn_rate, random_state):
        """Escalated tickets should never exceed total tickets."""
        df = generate_customer_data(
            num_customers=num_customers, num_months=num_months, churn_rate=churn_rate, random_state=random_state
        )

        assert (df["escalated_tickets"] <= df["total_tickets"]).all(), "Escalated tickets exceed total tickets"

    @given(
        num_customers=st.integers(min_value=5, max_value=100),
        num_months=st.integers(min_value=3, max_value=24),
        churn_rate=st.floats(min_value=0.0, max_value=1.0),
        random_state=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_enabled_channels_valid(self, num_customers, num_months, churn_rate, random_state):
        """Enabled channels should contain only valid channel names."""
        df = generate_customer_data(
            num_customers=num_customers, num_months=num_months, churn_rate=churn_rate, random_state=random_state
        )

        valid_channels = {"web", "card", "agent", "ivr", "apple_pay"}

        for channels_str in df["enabled_channels"]:
            channels = set(channels_str.split("|"))
            assert channels.issubset(valid_channels), f"Invalid channels found: {channels - valid_channels}"

    @given(
        num_customers=st.integers(min_value=5, max_value=100),
        num_months=st.integers(min_value=3, max_value=24),
        churn_rate=st.floats(min_value=0.0, max_value=1.0),
        random_state=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_customer_churns_at_most_once(self, num_customers, num_months, churn_rate, random_state):
        """Each customer should churn at most once."""
        df = generate_customer_data(
            num_customers=num_customers, num_months=num_months, churn_rate=churn_rate, random_state=random_state
        )

        churn_counts = df.groupby("account_id")["churned"].sum()
        assert (churn_counts <= 1).all(), "Customer churned more than once"

    @given(
        num_customers=st.integers(min_value=5, max_value=100),
        num_months=st.integers(min_value=3, max_value=24),
        churn_rate=st.floats(min_value=0.0, max_value=1.0),
        random_state=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_no_records_after_churn(self, num_customers, num_months, churn_rate, random_state):
        """Customer should have no records after churning."""
        df = generate_customer_data(
            num_customers=num_customers, num_months=num_months, churn_rate=churn_rate, random_state=random_state
        )

        # For customers who churned, verify no records after churn month
        churned_customers = df[df["churned"] == 1]["account_id"].unique()

        for customer in churned_customers:
            customer_df = df[df["account_id"] == customer].sort_values("month")
            churn_idx = customer_df[customer_df["churned"] == 1].index[0]
            # Churn record should be the last record
            assert churn_idx == customer_df.index[-1], f"Customer {customer} has records after churning"
