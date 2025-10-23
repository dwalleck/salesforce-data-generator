# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based synthetic data generator for customer churn prediction. The project generates realistic customer lifecycle data with temporal trends and various churn patterns.

## Development Commands

This project uses `uv` for dependency management.

**Install dependencies:**
```bash
uv sync
```

**Run the generator:**
```bash
uv run python synthetic_generator.py
```

**Interactive Python shell with dependencies:**
```bash
uv run python
```

## Architecture

### Core Module: `synthetic_generator.py`

The codebase consists of a single-module architecture with three main components:

1. **`generate_customer_data()`** - Main entry point that orchestrates the data generation process
   - Creates customer cohorts (churned vs stable)
   - Assigns churn patterns to customers who will churn
   - Generates monthly time-series records for each customer
   - Returns a pandas DataFrame with complete customer history

2. **`_get_churn_month()`** - Determines when a customer will churn based on their cohort:
   - `early_churn`: Months 1-3 (poor onboarding)
   - `renewal_churn`: Months 11-13 (contract renewal issues)
   - `gradual_disengagement`: Month 6+ (slow decline)
   - `service_issues`: Month 6+ (support ticket spike)
   - `price_sensitive`: Month 6+ (revenue decline)
   - `stable`: Never churns

3. **`_generate_month_metrics()`** - Creates realistic monthly metrics with cohort-specific patterns:
   - Applies declining trends 3 months before churn
   - Injects realistic anomalies (ticket spikes, reduced engagement, revenue drops)
   - Generates features: transactions, revenue, tickets, escalations, late payments, channels, self-service percentage, contact recency, resolution time

### Data Model

Each record represents one customer-month with these fields:
- `account_id`: Unique customer identifier (ACC-XXXX)
- `month`: Record date (YYYY-MM-DD format)
- `current_month_transactions`: Transaction count
- `current_month_revenue`: Revenue amount
- `total_tickets`: Support tickets opened
- `escalated_tickets`: Escalated support cases
- `late_payments`: Number of late payments
- `enabled_channels`: Count of active communication channels
- `self_service_percentage`: Self-service adoption rate
- `last_touchbase_date`: Most recent customer contact (ISO format)
- `average_resolution_time_hours`: Support resolution time
- `churned`: Binary flag (1 = customer churned this month)

### Key Design Patterns

**Churn Signal Injection**: The generator creates realistic leading indicators by applying cohort-specific degradation patterns in the 3 months preceding churn. This ensures the synthetic data contains learnable patterns for ML models.

**Temporal Consistency**: Each customer has a baseline characteristic set that persists across months with controlled variance, ensuring realistic longitudinal data.

**Configurable Parameters**: All key aspects are parameterized (num_customers, num_months, churn_rate, random_state) for flexible dataset generation.
