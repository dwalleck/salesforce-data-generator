"""Enhanced synthetic data generation for churn prediction."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd


class Channel(Enum):
    """Available communication channels for customers."""

    WEB = "web"
    CARD = "card"
    AGENT = "agent"
    IVR = "ivr"
    APPLE_PAY = "apple_pay"


class ChurnCohort(Enum):
    """Churn cohort types with associated month range calculation logic."""

    EARLY_CHURN = "early_churn"  # Poor onboarding, churns months 1-3
    RENEWAL_CHURN = "renewal_churn"  # Contract renewal issues, churns months 11-13
    GRADUAL_DISENGAGEMENT = "gradual_disengagement"  # Slow decline, churns month 6+
    SERVICE_ISSUES = "service_issues"  # Support ticket spike, churns month 6+
    PRICE_SENSITIVE = "price_sensitive"  # Revenue decline, churns month 6+
    STABLE = "stable"  # Never churns

    def get_churn_month(self, num_months: int) -> int:
        """
        Calculate the churn month for this cohort.

        Args:
            num_months: Total number of months in the dataset

        Returns:
            Month index (0-based) when churn occurs, or num_months if stable
        """
        if self == ChurnCohort.EARLY_CHURN:
            return np.random.randint(1, min(4, num_months))  # Month 1-3
        elif self == ChurnCohort.RENEWAL_CHURN:
            # Churn between months 11-13, but ensure we don't exceed num_months
            return np.random.randint(min(11, num_months - 1), min(14, num_months))
        elif self in [ChurnCohort.GRADUAL_DISENGAGEMENT, ChurnCohort.SERVICE_ISSUES, ChurnCohort.PRICE_SENSITIVE]:
            # Later churn (month 6+), but ensure valid range
            return np.random.randint(min(6, num_months - 1), num_months)
        else:  # STABLE
            return num_months  # Never churns


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation parameters."""

    # Baseline customer characteristics (set once per customer at creation)
    baseline_transactions_min: int = 50  # Minimum monthly transactions for a customer
    baseline_transactions_max: int = 300  # Maximum monthly transactions for a customer
    baseline_revenue_multiplier_min: float = 50.0  # Minimum dollars per transaction
    baseline_revenue_multiplier_max: float = 500.0  # Maximum dollars per transaction
    baseline_tickets_lambda: int = 2  # Lambda for Poisson distribution of support tickets (~2 tickets/month avg)

    # Monthly variance factors (applied each month to simulate natural fluctuation)
    transaction_variance_min: float = 0.9  # Minimum monthly variance multiplier for transactions (90% of baseline)
    transaction_variance_max: float = 1.1  # Maximum monthly variance multiplier for transactions (110% of baseline)
    revenue_variance_min: float = 0.9  # Minimum monthly variance multiplier for revenue (90% of baseline)
    revenue_variance_max: float = 1.1  # Maximum monthly variance multiplier for revenue (110% of baseline)

    # Ticket and service parameters (baseline healthy customer behavior)
    ticket_variance_range: tuple[int, int] = (-1, 2)  # Random adjustment to baseline tickets each month (±1)
    days_since_contact_min: int = 1  # Minimum days since last customer contact (recent engagement)
    days_since_contact_max: int = 30  # Maximum days since last customer contact (less engaged)
    resolution_time_min: float = 20.0  # Minimum average ticket resolution time in hours
    resolution_time_max: float = 60.0  # Maximum average ticket resolution time in hours

    # Channel enablement probabilities (independent probabilities for each channel)
    channel_web_prob: float = 0.80  # Probability that web channel is enabled (80%)
    channel_card_prob: float = 0.70  # Probability that card channel is enabled (70%)
    channel_agent_prob: float = 0.60  # Probability that agent channel is enabled (60%)
    channel_ivr_prob: float = 0.40  # Probability that IVR channel is enabled (40%)
    channel_apple_pay_prob: float = 0.20  # Probability that Apple Pay is enabled (20%, requires web)

    self_service_pct_min: float = 30.0  # Minimum percentage of issues resolved via self-service
    self_service_pct_max: float = 80.0  # Maximum percentage of issues resolved via self-service
    escalation_probability: float = 0.2  # Probability that a support ticket gets escalated (20%)

    # Late payment parameters (using binomial distribution)
    late_payment_trials: int = 3  # Number of payment opportunities per month
    late_payment_normal_prob: float = 0.1  # Probability of late payment for healthy customers (10%)
    late_payment_price_sensitive_prob: float = 0.4  # Probability of late payment for price-sensitive churners (40%)

    # Churn signal parameters for "gradual_disengagement" cohort (applied 3 months before churn)
    gradual_disengagement_decline_rate: float = 0.15  # Rate of decline per month (15%)
    gradual_disengagement_ticket_decline: float = 0.2  # Ticket decline rate per month (20%)
    gradual_disengagement_contact_delay_per_month: int = 10  # Days added to contact delay per month

    # Churn signal parameters for "service_issues" cohort (applied 3 months before churn)
    service_issues_ticket_multiplier: int = 3  # Ticket volume multiplier (3x baseline)
    service_issues_resolution_multiplier: float = 1.5  # Resolution time multiplier (1.5x baseline)
    service_issues_contact_delay_per_month: int = 20  # Days added to contact delay per month
    service_issues_max_contact_delay: int = 90  # Maximum days since last contact (cap)

    # Churn signal parameters for "price_sensitive" cohort (applied 3 months before churn)
    price_sensitive_revenue_min_factor: float = 0.7  # Minimum revenue factor at churn month (70% of baseline)
    price_sensitive_revenue_decline_rate: float = 0.1  # Revenue decline rate per month approaching churn (10%)

    # Churn signal parameters for "early_churn" cohort (poor onboarding)
    early_churn_ticket_multiplier: int = 2  # Ticket volume multiplier (2x baseline)
    early_churn_self_service_factor: float = 0.5  # Self-service reduction factor (50%)
    early_churn_channel_removal_prob: float = 0.5  # Channel removal probability (50% per channel)
    early_churn_min_self_service: float = 10.0  # Minimum self-service percentage floor (10%)
    early_churn_min_channels: int = 1  # Minimum number of enabled channels (at least 1)


def _generate_channels(config: GeneratorConfig) -> list[Channel]:
    """
    Generate a list of enabled channels based on configuration probabilities.

    Enforces the constraint that Apple Pay requires Web to be enabled.

    Args:
        config: Generator configuration with channel probabilities

    Returns:
        List of enabled Channel enum values
    """
    channels = []

    # Independently check each channel based on probability
    if np.random.random() < config.channel_web_prob:
        channels.append(Channel.WEB)
    if np.random.random() < config.channel_card_prob:
        channels.append(Channel.CARD)
    if np.random.random() < config.channel_agent_prob:
        channels.append(Channel.AGENT)
    if np.random.random() < config.channel_ivr_prob:
        channels.append(Channel.IVR)
    if np.random.random() < config.channel_apple_pay_prob:
        channels.append(Channel.APPLE_PAY)

    # Enforce constraint: Apple Pay requires Web
    if Channel.APPLE_PAY in channels and Channel.WEB not in channels:
        channels.append(Channel.WEB)

    # Ensure at least one channel is enabled
    if not channels:
        channels.append(Channel.WEB)  # Default to web if no channels selected

    return channels


def generate_customer_data(
    num_customers: int = 1000,
    num_months: int = 12,
    churn_rate: float = 0.15,
    random_state: int = 42,
    config: GeneratorConfig | None = None,
) -> pd.DataFrame:
    """
    Generate realistic synthetic customer data with temporal trends.

    Args:
        num_customers: Number of unique customers
        num_months: Number of months of history per customer
        churn_rate: Target churn rate (0-1)
        random_state: Random seed for reproducibility
        config: Optional configuration object for generator parameters

    Returns:
        DataFrame with customer records

    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if num_customers <= 0:
        raise ValueError(f"num_customers must be positive, got {num_customers}")
    if num_months < 3:
        raise ValueError(f"num_months must be at least 3 for churn patterns, got {num_months}")
    if not 0 <= churn_rate <= 1:
        raise ValueError(f"churn_rate must be between 0 and 1, got {churn_rate}")
    if random_state < 0:
        raise ValueError(f"random_state must be non-negative, got {random_state}")

    if config is None:
        config = GeneratorConfig()

    np.random.seed(random_state)

    # Initialize column lists for efficient DataFrame construction
    columns = {
        "account_id": [],
        "month": [],
        "current_month_transactions": [],
        "current_month_revenue": [],
        "total_tickets": [],
        "escalated_tickets": [],
        "late_payments": [],
        "enabled_channels": [],
        "self_service_percentage": [],
        "last_touchbase_date": [],
        "average_resolution_time_hours": [],
        "churned": [],
    }
    start_date = datetime(2024, 1, 1)

    # Determine which customers will churn
    num_churned = int(num_customers * churn_rate)
    churned_customers = set(np.random.choice(num_customers, num_churned, replace=False))

    for customer_id in range(num_customers):
        will_churn = customer_id in churned_customers

        # Customer baseline characteristics
        baseline_transactions = np.random.randint(config.baseline_transactions_min, config.baseline_transactions_max)
        baseline_revenue = baseline_transactions * np.random.uniform(
            config.baseline_revenue_multiplier_min,
            config.baseline_revenue_multiplier_max,
        )
        baseline_tickets = np.random.poisson(config.baseline_tickets_lambda)
        baseline_channels = _generate_channels(config)

        # Churn cohort assignment
        if will_churn:
            cohorts = [
                ChurnCohort.EARLY_CHURN,
                ChurnCohort.RENEWAL_CHURN,
                ChurnCohort.GRADUAL_DISENGAGEMENT,
                ChurnCohort.SERVICE_ISSUES,
                ChurnCohort.PRICE_SENSITIVE,
            ]
            churn_cohort = cohorts[np.random.randint(0, len(cohorts))]
            churn_month = churn_cohort.get_churn_month(num_months)
        else:
            churn_cohort = ChurnCohort.STABLE
            churn_month = None

        # Generate monthly records
        for month_offset in range(num_months):
            month_date = start_date + pd.DateOffset(months=month_offset)
            is_churn_month = churn_month == month_offset

            # Apply cohort-specific patterns
            metrics = _generate_month_metrics(
                baseline_transactions=baseline_transactions,
                baseline_revenue=baseline_revenue,
                baseline_tickets=baseline_tickets,
                baseline_channels=baseline_channels,
                month_offset=month_offset,
                churn_cohort=churn_cohort,
                churn_month=churn_month,
                config=config,
            )

            # Append values to column lists for efficient DataFrame construction
            columns["account_id"].append(f"ACC-{customer_id:04d}")
            columns["month"].append(month_date.strftime("%Y-%m-%d"))
            columns["current_month_transactions"].append(metrics["transactions"])
            columns["current_month_revenue"].append(metrics["revenue"])
            columns["total_tickets"].append(metrics["tickets"])
            columns["escalated_tickets"].append(metrics["escalated_tickets"])
            columns["late_payments"].append(metrics["late_payments"])
            columns["enabled_channels"].append("|".join(ch.value for ch in metrics["channels"]))
            columns["self_service_percentage"].append(metrics["self_service_pct"])
            columns["last_touchbase_date"].append(
                (month_date - timedelta(days=metrics["days_since_contact"])).isoformat()
            )
            columns["average_resolution_time_hours"].append(metrics["resolution_time"])
            columns["churned"].append(1 if is_churn_month else 0)

            # Stop generating records after churn
            if is_churn_month:
                break

    return pd.DataFrame(columns)


def _apply_gradual_disengagement_pattern(
    transactions: float,
    revenue: float,
    tickets: int,
    days_since_contact: int,
    months_until_churn: int,
    config: GeneratorConfig,
) -> tuple[float, float, int, int]:
    """
    Apply gradual disengagement churn pattern.

    Simulates customer slowly reducing engagement over 3 months before churn.
    Transactions and revenue decline, tickets decrease, contact becomes less frequent.

    Args:
        transactions: Base transaction count for the month
        revenue: Base revenue for the month
        tickets: Base ticket count for the month
        days_since_contact: Base days since last contact
        months_until_churn: Number of months remaining until churn (3, 2, 1, or 0)
        config: Configuration with pattern parameters

    Returns:
        Tuple of (adjusted_transactions, adjusted_revenue, adjusted_tickets,
                  adjusted_days_since_contact)
    """
    # Gradual decline over 3 months
    decline_factor = 1 - (config.gradual_disengagement_decline_rate * (3 - months_until_churn))
    transactions *= decline_factor
    revenue *= decline_factor
    tickets = max(
        0,
        int(tickets * (1 - config.gradual_disengagement_ticket_decline * (3 - months_until_churn))),
    )
    days_since_contact += int(config.gradual_disengagement_contact_delay_per_month * (3 - months_until_churn))
    return transactions, revenue, tickets, days_since_contact


def _apply_service_issues_pattern(
    tickets: int, resolution_time: float, days_since_contact: int, months_until_churn: int, config: GeneratorConfig
) -> tuple[int, float, int]:
    """
    Apply service issues churn pattern.

    Simulates customer experiencing poor service quality. Tickets spike,
    resolution times increase, and customer contact becomes infrequent.

    Args:
        tickets: Base ticket count for the month
        resolution_time: Base resolution time in hours
        days_since_contact: Base days since last contact
        months_until_churn: Number of months remaining until churn
        config: Configuration with pattern parameters

    Returns:
        Tuple of (adjusted_tickets, adjusted_resolution_time, adjusted_days_since_contact)
    """
    # Spike in tickets and escalations
    tickets = int(tickets * config.service_issues_ticket_multiplier)
    resolution_time *= config.service_issues_resolution_multiplier
    days_since_contact = min(
        config.service_issues_max_contact_delay,
        days_since_contact + config.service_issues_contact_delay_per_month * (3 - months_until_churn),
    )
    return tickets, resolution_time, days_since_contact


def _apply_price_sensitive_pattern(revenue: float, months_until_churn: int, config: GeneratorConfig) -> float:
    """
    Apply price-sensitive churn pattern.

    Simulates customer reducing spend while maintaining transaction volume.
    Revenue declines while transactions stay stable. Late payments increase
    (handled separately in _generate_month_metrics).

    Args:
        revenue: Base revenue for the month
        months_until_churn: Number of months remaining until churn
        config: Configuration with pattern parameters

    Returns:
        Adjusted revenue amount
    """
    # Revenue decline but transactions stable (late payments handled separately)
    revenue *= (
        config.price_sensitive_revenue_min_factor + config.price_sensitive_revenue_decline_rate * months_until_churn
    )
    return revenue


def _apply_early_churn_pattern(
    tickets: int,
    self_service_pct: float,
    channels: list[Channel],
    baseline_channels: list[Channel],
    config: GeneratorConfig,
) -> tuple[int, float, list[Channel]]:
    """
    Apply early churn pattern (poor onboarding).

    Simulates customer struggling with onboarding. High support ticket volume,
    low self-service adoption, and channel abandonment.

    Args:
        tickets: Base ticket count for the month
        self_service_pct: Base self-service percentage
        channels: Current enabled channels
        baseline_channels: Original enabled channels for the customer
        config: Configuration with pattern parameters

    Returns:
        Tuple of (adjusted_tickets, adjusted_self_service_pct, adjusted_channels)
    """
    # Poor onboarding - high tickets, low adoption
    tickets = int(tickets * config.early_churn_ticket_multiplier)
    self_service_pct = max(
        config.early_churn_min_self_service,
        self_service_pct * config.early_churn_self_service_factor,
    )
    # Randomly remove channels (50% probability per channel)
    channels = [ch for ch in channels if np.random.random() > config.early_churn_channel_removal_prob]
    # Ensure at least one channel remains
    if len(channels) < config.early_churn_min_channels:
        channels = baseline_channels[: config.early_churn_min_channels]

    return tickets, self_service_pct, channels


def _generate_month_metrics(
    baseline_transactions: int,
    baseline_revenue: float,
    baseline_tickets: int,
    baseline_channels: list[Channel],
    month_offset: int,
    churn_cohort: ChurnCohort,
    churn_month: int | None,
    config: GeneratorConfig,
) -> dict[str, int | float | list[Channel]]:
    """
    Generate realistic monthly metrics based on churn cohort.

    Applies base randomness to customer characteristics, then applies cohort-specific
    churn patterns during the 3 months leading up to churn.

    Args:
        baseline_transactions: Customer's base monthly transaction count
        baseline_revenue: Customer's base monthly revenue
        baseline_tickets: Customer's base monthly support tickets
        baseline_channels: Customer's baseline enabled channels
        month_offset: Current month index (0-based)
        churn_cohort: Type of churn pattern to apply
        churn_month: Month when customer will churn (None if stable)
        config: Configuration with variance and pattern parameters

    Returns:
        Dictionary with keys: transactions, revenue, tickets, escalated_tickets,
        late_payments, channels, self_service_pct, days_since_contact, resolution_time
    """
    metrics = {}

    # Base values with natural month-to-month variance
    # Apply randomness first to simulate normal business fluctuations,
    # then cohort patterns are applied on top to create churn signals
    transactions = baseline_transactions * np.random.uniform(
        config.transaction_variance_min, config.transaction_variance_max
    )
    revenue = baseline_revenue * np.random.uniform(config.revenue_variance_min, config.revenue_variance_max)
    tickets = max(0, baseline_tickets + np.random.randint(*config.ticket_variance_range))
    days_since_contact = np.random.randint(config.days_since_contact_min, config.days_since_contact_max)
    resolution_time = np.random.uniform(config.resolution_time_min, config.resolution_time_max)
    # Start with baseline channels, will be modified by churn patterns below
    channels = baseline_channels.copy()
    self_service_pct = np.random.uniform(config.self_service_pct_min, config.self_service_pct_max)

    # Apply cohort-specific patterns during the 3-month pre-churn window
    # This creates learnable patterns for ML models by introducing progressive
    # degradation in customer behavior metrics leading up to churn
    if churn_month is not None and month_offset >= churn_month - 3:
        months_until_churn = churn_month - month_offset  # 3, 2, 1, or 0

        if churn_cohort == ChurnCohort.GRADUAL_DISENGAGEMENT:
            transactions, revenue, tickets, days_since_contact = _apply_gradual_disengagement_pattern(
                transactions, revenue, tickets, days_since_contact, months_until_churn, config
            )

        elif churn_cohort == ChurnCohort.SERVICE_ISSUES:
            tickets, resolution_time, days_since_contact = _apply_service_issues_pattern(
                tickets, resolution_time, days_since_contact, months_until_churn, config
            )

        elif churn_cohort == ChurnCohort.PRICE_SENSITIVE:
            revenue = _apply_price_sensitive_pattern(revenue, months_until_churn, config)

        elif churn_cohort == ChurnCohort.EARLY_CHURN:
            tickets, self_service_pct, channels = _apply_early_churn_pattern(
                tickets, self_service_pct, channels, baseline_channels, config
            )

    # Finalize metrics with bounds checking and formatting
    metrics["transactions"] = int(max(0, transactions))
    metrics["revenue"] = round(max(0, revenue), 2)  # Round to 2 decimal places for currency
    metrics["tickets"] = int(max(0, tickets))

    # Escalated tickets: binomial distribution, capped at total tickets
    metrics["escalated_tickets"] = int(
        min(
            metrics["tickets"],
            np.random.binomial(metrics["tickets"], config.escalation_probability),
        )
    )

    # Late payments: price-sensitive customers have 4x higher probability (40% vs 10%)
    # This creates a strong signal for the price_sensitive churn cohort
    metrics["late_payments"] = (
        np.random.binomial(config.late_payment_trials, config.late_payment_normal_prob)
        if churn_cohort != ChurnCohort.PRICE_SENSITIVE
        else np.random.binomial(config.late_payment_trials, config.late_payment_price_sensitive_prob)
    )
    metrics["channels"] = channels
    metrics["self_service_pct"] = min(100, max(0, self_service_pct))
    metrics["days_since_contact"] = min(365, max(0, int(days_since_contact)))
    metrics["resolution_time"] = max(0, resolution_time)

    return metrics


def main():
    """Command-line interface for synthetic data generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic customer data for churn prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default dataset (1000 customers, 12 months)
  python synthetic_generator.py -o output.csv

  # Generate larger dataset with higher churn rate
  python synthetic_generator.py -n 10000 -m 24 -c 0.25 -o data.csv

  # Specify custom random seed for reproducibility
  python synthetic_generator.py -n 5000 -r 123 -o data.csv
        """,
    )

    parser.add_argument(
        "-n", "--num-customers", type=int, default=1000, help="Number of unique customers to generate (default: 1000)"
    )

    parser.add_argument(
        "-m",
        "--num-months",
        type=int,
        default=12,
        help="Number of months of history per customer (default: 12, minimum: 3)",
    )

    parser.add_argument(
        "-c", "--churn-rate", type=float, default=0.15, help="Target churn rate between 0 and 1 (default: 0.15)"
    )

    parser.add_argument(
        "-r", "--random-seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path (CSV format)")

    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet", "json"],
        default="csv",
        help="Output file format (default: csv)",
    )

    args = parser.parse_args()

    # Generate data
    print("Generating synthetic customer data...")
    print(f"  Customers: {args.num_customers}")
    print(f"  Months: {args.num_months}")
    print(f"  Churn rate: {args.churn_rate:.1%}")
    print(f"  Random seed: {args.random_seed}")

    try:
        df = generate_customer_data(
            num_customers=args.num_customers,
            num_months=args.num_months,
            churn_rate=args.churn_rate,
            random_state=args.random_seed,
        )

        # Save output
        if args.format == "csv":
            df.to_csv(args.output, index=False)
        elif args.format == "parquet":
            df.to_parquet(args.output, index=False)
        elif args.format == "json":
            df.to_json(args.output, orient="records", indent=2)

        print(f"\n✓ Successfully generated {len(df)} records")
        print(f"✓ Saved to: {args.output}")

        # Summary statistics
        churned_customers = df[df["churned"] == 1]["account_id"].nunique()
        total_customers = df["account_id"].nunique()
        actual_churn_rate = churned_customers / total_customers

        print("\nSummary:")
        print(f"  Total customers: {total_customers}")
        print(f"  Churned customers: {churned_customers}")
        print(f"  Actual churn rate: {actual_churn_rate:.1%}")

    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
