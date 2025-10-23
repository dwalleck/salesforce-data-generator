"""Feature engineering for customer churn prediction.

Generates temporal features from customer transaction data including
rolling aggregates, trends, and volatility metrics.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _add_rolling_features(
    df: pd.DataFrame, col: str, windows: dict[str, tuple[int, int]], div_by_zero_offset: float = 1.0
) -> pd.DataFrame:
    """
    Add rolling window temporal features for a column.

    Generates sum, average, max, min, trend, and volatility features for each
    specified time window. Trend is calculated as current value divided by
    window average. Volatility is standard deviation over the window.

    Args:
        df: DataFrame (must be sorted by account_id, month)
        col: Column name to generate features for (must exist in df)
        windows: Dict mapping window names to (months, min_periods) tuples.
                 e.g., {'90d': (3, 1), '180d': (6, 3)} creates features for
                 90-day (3 months) and 180-day (6 months) windows
        div_by_zero_offset: Offset added to denominators in trend calculations
                           to prevent division by zero (default: 1.0)

    Returns:
        DataFrame with additional rolling feature columns. For each window,
        adds: {col}_{window}_sum, {col}_{window}_avg, {col}_{window}_max,
        {col}_{window}_min, {col}_trend_{window}, {col}_volatility_{window}
    """
    for window_name, (months, min_periods) in windows.items():
        # Compute rolling object once and apply multiple aggregations
        rolling = df.groupby("account_id")[col].rolling(months, min_periods=min_periods)

        # Apply all basic aggregations in one pass
        agg_results = rolling.agg(["sum", "mean", "max", "min"]).reset_index(0, drop=True)

        df[f"{col}_{window_name}_sum"] = agg_results["sum"]
        df[f"{col}_{window_name}_avg"] = agg_results["mean"]
        df[f"{col}_{window_name}_max"] = agg_results["max"]
        df[f"{col}_{window_name}_min"] = agg_results["min"]

        # Trend: current vs window average
        df[f"{col}_trend_{window_name}"] = df[col] / (df[f"{col}_{window_name}_avg"] + div_by_zero_offset)

        # Volatility (standard deviation) - requires at least 2 periods
        if min_periods >= 2 or months > 1:
            df[f"{col}_volatility_{window_name}"] = (
                df.groupby("account_id")[col]
                .rolling(months, min_periods=max(2, min_periods))
                .std()
                .reset_index(0, drop=True)
            )

    return df


def _add_change_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Add month-over-month change and acceleration features.

    Percent change measures the relative change from previous month.
    Acceleration measures the rate of change of percent change (second derivative).

    Args:
        df: DataFrame (must be sorted by account_id, month)
        col: Column name to generate features for (must exist in df)

    Returns:
        DataFrame with two additional columns:
        - {col}_percent_change: Month-over-month percentage change
        - {col}_acceleration: Change in the rate of change
    """
    # Percent change (month-over-month)
    df[f"{col}_percent_change"] = df.groupby("account_id")[col].pct_change()

    # Acceleration (rate of change of percent change)
    df[f"{col}_acceleration"] = df.groupby("account_id")[f"{col}_percent_change"].diff()

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive temporal features for customer churn prediction.

    Features generated:
    - Channel features: has_<channel>, number_of_channels, channel trends, abandonment
    - Customer metadata: days_since_last_contact, account_age_months
    - Temporal aggregates: 90d/180d sums, averages, max, min, trends, volatility
    - Change metrics: percent_change, acceleration
    - Service quality: self-service adoption, resolution time patterns

    Args:
        df: DataFrame with 'account_id', 'month', 'current_month_revenue',
            'current_month_transactions', 'enabled_channels', 'last_touchbase_date',
            'late_payments', 'escalated_tickets', 'self_service_percentage',
            'average_resolution_time_hours' columns

    Returns:
        DataFrame with comprehensive feature columns

    Raises:
        ValueError: If required columns are missing or DataFrame is empty
    """
    # Validate input
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    required_columns = [
        "account_id",
        "month",
        "current_month_revenue",
        "current_month_transactions",
        "enabled_channels",
        "last_touchbase_date",
        "late_payments",
        "escalated_tickets",
        "self_service_percentage",
        "average_resolution_time_hours",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.sort_values(["account_id", "month"]).copy()

    # ========== Channel Features ==========
    channels = ["web", "card", "agent", "ivr", "apple_pay"]
    for channel in channels:
        df[f"has_{channel}"] = df["enabled_channels"].str.contains(channel, na=False).astype(int)

    df["number_of_channels"] = df[[f"has_{ch}" for ch in channels]].sum(axis=1)

    # ========== Customer Metadata Features ==========
    # Convert last_touchbase_date to days since last contact
    df["month_date"] = pd.to_datetime(df["month"])
    df["last_touchbase_datetime"] = pd.to_datetime(df["last_touchbase_date"])
    df["days_since_last_contact"] = (df["month_date"] - df["last_touchbase_datetime"]).dt.days

    # Calculate account age in months (tenure)
    df["first_month"] = df.groupby("account_id")["month_date"].transform("min")
    df["account_age_months"] = (df["month_date"].dt.year - df["first_month"].dt.year) * 12 + (
        df["month_date"].dt.month - df["first_month"].dt.month
    )

    # Clean up temporary columns
    df = df.drop(columns=["month_date", "last_touchbase_datetime", "first_month"])

    # ========== Temporal Features Configuration ==========
    # Use consistent time windows across all metrics to capture short and long-term patterns
    # 90d (3 months, min 1): Recent trends, captures immediate pre-churn signals
    # 180d (6 months, min 3): Long-term patterns, captures seasonal/sustained changes
    windows = {"90d": (3, 1), "180d": (6, 3)}

    # ========== Revenue & Transaction Temporal Features ==========
    for col in ["current_month_revenue", "current_month_transactions"]:
        df = _add_rolling_features(df, col, windows, div_by_zero_offset=1.0)
        df = _add_change_features(df, col)

        # Additional comparison features
        df[f"{col}_30d"] = df[col]  # Current month value

        # Recent vs historical comparison: compares current month to previous 2 months
        # This helps detect sudden changes (e.g., spike or drop vs recent baseline)
        # shift(1) gets previous month's values, then rolling(2).sum() sums 2 months
        # Result: current_month / (month_-1 + month_-2)
        prev_60d = df.groupby("account_id")[col].shift(1).rolling(2, min_periods=1).sum().reset_index(0, drop=True)
        df[f"{col}_recent_vs_historical"] = df[col] / (prev_60d + 1)

    # ========== Support & Payment Temporal Features ==========
    for col in ["late_payments", "escalated_tickets"]:
        df = _add_rolling_features(df, col, windows, div_by_zero_offset=0.1)
        df = _add_change_features(df, col)

    # ========== Service Quality Temporal Features ==========
    # Self-service percentage: lower is worse (poor adoption)
    df = _add_rolling_features(df, "self_service_percentage", windows, div_by_zero_offset=1.0)
    df = _add_change_features(df, "self_service_percentage")

    # Average resolution time: higher is worse (slow support)
    df = _add_rolling_features(df, "average_resolution_time_hours", windows, div_by_zero_offset=1.0)
    df = _add_change_features(df, "average_resolution_time_hours")

    # ========== Channel Temporal Features ==========
    df["number_of_channels_avg_90d"] = (
        df.groupby("account_id")["number_of_channels"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    )
    df["channel_trend"] = df["number_of_channels"] / (df["number_of_channels_avg_90d"] + 0.01)
    df["max_channels_ever"] = df.groupby("account_id")["number_of_channels"].cummax()
    df["channels_dropped"] = df["max_channels_ever"] - df["number_of_channels"]

    # ========== Fill Missing Values ==========
    # NaNs appear in derived features (percent_change, acceleration) for early months
    # where insufficient history exists. Fill with 0 to represent "no change yet".
    # Division by zero is already prevented via offset parameters in all calculations.
    df = df.fillna(0)

    return df


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV, JSON, or Parquet file.

    Args:
        file_path: Path to input file. Must have .csv, .json, or .parquet extension

    Returns:
        DataFrame containing the loaded data

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If file format is not supported (not .csv, .json, or .parquet)
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif path.suffix.lower() == ".json":
        return pd.read_json(file_path)
    elif path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv, .json, or .parquet")


def save_data(df: pd.DataFrame, file_path: str, output_format: str | None = None):
    """
    Save DataFrame to file in specified format.

    Args:
        df: DataFrame to save
        file_path: Output file path
        output_format: Output format ('csv', 'json', or 'parquet').
                      If None, inferred from file_path extension

    Raises:
        ValueError: If output_format is not 'csv', 'json', or 'parquet'
    """
    path = Path(file_path)

    # Determine format from extension if not specified
    if output_format is None:
        output_format = path.suffix.lower().lstrip(".")

    if output_format == "csv":
        df.to_csv(file_path, index=False)
    elif output_format == "json":
        df.to_json(file_path, orient="records", indent=2)
    elif output_format == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main() -> int:
    """
    Command-line interface for feature engineering.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Generate temporal features for customer churn prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate features from CSV
  python feature_engineer.py data.csv

  # Specify custom output file
  python feature_engineer.py data.csv -o enriched_data.csv

  # Convert to JSON format
  python feature_engineer.py data.csv --format json
        """,
    )

    parser.add_argument("input_file", type=str, help="Input data file (CSV, JSON, or Parquet)")

    parser.add_argument(
        "-o", "--output", type=str, help="Output file path (default: adds _features suffix to input filename)"
    )

    parser.add_argument(
        "--format", type=str, choices=["csv", "json", "parquet"], help="Output file format (default: same as input)"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"{input_path.stem}_features{input_path.suffix}"

    print(f"Loading data from {args.input_file}...")
    try:
        df = load_data(args.input_file)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return 1

    print(f"  Loaded {len(df)} records")
    print(f"  Unique customers: {df['account_id'].nunique()}")

    # Validate required columns (must match generate_features() requirements)
    required_cols = [
        "account_id",
        "month",
        "current_month_revenue",
        "current_month_transactions",
        "enabled_channels",
        "last_touchbase_date",
        "late_payments",
        "escalated_tickets",
        "self_service_percentage",
        "average_resolution_time_hours",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Error: Missing required columns: {missing_cols}")
        return 1

    # Generate features
    print("\nGenerating temporal features...")
    original_cols = len(df.columns)

    try:
        df = generate_features(df)
    except Exception as e:
        print(f"✗ Error generating features: {e}")
        return 1

    new_cols = len(df.columns) - original_cols
    print(f"  Added {new_cols} feature columns")

    # Save output
    print(f"\nSaving to {output_path}...")
    try:
        save_data(df, str(output_path), args.format)
    except Exception as e:
        print(f"✗ Error saving data: {e}")
        return 1

    print(f"✓ Successfully saved {len(df)} records with {len(df.columns)} total columns")

    # Summary
    print("\nSummary:")
    print(f"  Input file: {args.input_file}")
    print(f"  Output file: {output_path}")
    print(f"  Features added: {new_cols}")
    print(f"  Total columns: {len(df.columns)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
