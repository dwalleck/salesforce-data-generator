# Actionable Insights Guide

This guide shows how to use the improved prediction system that provides **actionable insights**, not just binary predictions.

## 🎯 Quick Start

Instead of just "will this customer churn?", we now answer:
- **How likely** is this customer to churn? (Risk score)
- **Why** are they at risk? (Feature importance)
- **What** should we do about it? (Segmentation & actions)

## 📊 1. Risk Scores Instead of Binary Predictions

### Old Way (Binary - Not Actionable)
```bash
python predict.py --model models_3month/xgboost.joblib --input data/features.csv
# Output: Customer A: 0 (no churn), Customer B: 0 (no churn)
# Problem: Both look the same, but one might be 49% risk!
```

### New Way (Risk Scores - Actionable)
```bash
python predict.py --model models_3month/xgboost.joblib --input data/features.csv \
    --threshold 0.117 --output data/predictions_with_risk.csv
```

**Output includes:**
- `churn_risk_score`: Probability from 0-100%
- `risk_tier`: High Risk / Medium Risk / Low Risk / Very Safe
- `prediction`: Binary (1/0) based on optimal threshold

**Sample Output:**
```
account_id    churn_risk_score  risk_tier     prediction
ACC-0001      0.85             High Risk      1
ACC-0002      0.55             Medium Risk    1
ACC-0003      0.12             Very Safe      0
```

### Risk Tier Actions

| Risk Tier | Churn Probability | Action |
|-----------|------------------|--------|
| **High Risk** | 70%+ | Personal outreach from account manager, special discount |
| **Medium Risk** | 40-70% | Automated email campaign, product tips |
| **Low Risk** | 20-40% | Monitor, no immediate action |
| **Very Safe** | <20% | Upsell opportunity! |

## 🔍 2. Understanding WHY Customers Are at Risk

### Analyze Churn Drivers
```bash
python analyze_churn_drivers.py --input data/features.csv \
    --model models_3month/xgboost.joblib --top-n 15
```

**This reveals:**
- Which features drive churn (e.g., "low self-service adoption", "late payments")
- Feature categories (Support Quality, Payment Behavior, Engagement)
- Visual plots saved to `results_3month/xgboost_churn_drivers.png`

**Example Output:**
```
Top 10 Features Driving Churn:
============================================================
  self_service_percentage_90d_min         0.3322  [Self-Service]
  late_payments_180d_sum                  0.2413  [Payment Behavior]
  days_since_last_contact                 0.1473  [Engagement]
```

**Business Insight:** Focus retention efforts on:
1. Improving self-service tools/training
2. Addressing payment issues early
3. Increasing customer engagement frequency

## 👥 3. Customer Segmentation - Different Solutions for Different Problems

### Segment High-Risk Customers by Root Cause
```bash
python analyze_churn_drivers.py --input data/features.csv \
    --model models_3month/xgboost.joblib --segment --export-segments
```

**Identifies segments like:**

### Segment 1: Support Issues
**Characteristics:** High ticket volume, long resolution times
**Root Cause:** Poor service quality
**Action:** Assign dedicated support rep, proactive check-ins, prioritize tickets

### Segment 2: Declining Engagement
**Characteristics:** Low transaction volume, few enabled channels
**Root Cause:** Poor onboarding or feature adoption
**Action:** Onboarding campaign, product tutorials, feature adoption emails

### Segment 3: Payment Issues
**Characteristics:** Late payments
**Root Cause:** Financial constraints
**Action:** Flexible payment plans, financial hardship outreach, billing support

### Segment 4: Revenue Decline
**Characteristics:** Declining spend while maintaining transactions
**Root Cause:** Not seeing value
**Action:** Show ROI evidence, success stories, upsell value-add features

**Exported to:** `results_3month/segments/*.csv` (ready for CRM import)

## 🎬 Complete Workflow Example

### Step 1: Generate Predictions with Risk Scores
```bash
# Use optimal threshold from optimization
python optimize_threshold.py --input data/features.csv --compare-all

# Make predictions with that threshold
python predict.py --model models_3month/xgboost.joblib \
    --input data/features.csv \
    --threshold 0.117 \
    --output data/predictions_with_risk.csv
```

### Step 2: Analyze What Drives Churn
```bash
python analyze_churn_drivers.py --input data/features.csv \
    --model models_3month/xgboost.joblib \
    --segment \
    --export-segments
```

### Step 3: Take Action by Risk Tier

**High Risk Customers (70%+):**
```sql
-- Load into CRM
SELECT account_id, churn_risk_score
FROM predictions_with_risk
WHERE risk_tier = 'High Risk'
ORDER BY churn_risk_score DESC
```
→ Queue for personal outreach from account managers

**Medium Risk (40-70%):**
→ Add to automated email nurture campaign
→ Send product tips and best practices

**Very Safe (<20%):**
→ Identify upsell opportunities
→ Request referrals, case studies

### Step 4: Targeted Interventions by Segment

```python
# Example: Email campaign for declining engagement segment
import pandas as pd

segment_df = pd.read_csv('results_3month/segments/declining_engagement.csv')

for _, customer in segment_df.iterrows():
    if customer['churn_risk_score'] > 0.7:
        # Send personalized onboarding series
        send_email(
            to=customer['account_id'],
            template='onboarding_intensive',
            priority='high'
        )
    elif customer['churn_risk_score'] > 0.4:
        # Send product tips
        send_email(
            to=customer['account_id'],
            template='product_tips',
            priority='medium'
        )
```

## 📈 Measuring Success

### Before (Binary Predictions)
- "We predicted 150 customers would churn"
- **Problem:** Don't know which 150, why, or what to do

### After (Risk Scores + Segmentation)
- "We identified 23 High Risk customers (70%+ churn probability)"
  - 8 have support issues → Assigned dedicated rep
  - 7 have declining engagement → Sent onboarding series
  - 5 have payment issues → Offered flexible plans
  - 3 have revenue decline → Sent ROI case studies

- **Result:** Specific, measurable actions with clear next steps

## 💡 Key Insights from Your Data

Based on the analysis of your generated data:

**Top 3 Churn Drivers:**
1. **Self-Service Adoption (33%):** Low self-service percentage predicts churn
2. **Payment Behavior (24%):** Late payments are a strong signal
3. **Customer Engagement (15%):** Days since last contact matters

**Recommended Focus Areas:**
1. ✅ Improve self-service portal and documentation
2. ✅ Implement proactive payment issue detection
3. ✅ Increase customer touchpoint frequency

## 🔧 Advanced: Optimal Thresholds

The default 0.5 threshold is wrong for churn prediction!

```bash
# Find optimal threshold for your business needs
python optimize_threshold.py --input data/features.csv \
    --model models_3month/xgboost.joblib \
    --metric recall \
    --min-precision 0.5
```

**Threshold Trade-offs:**
- **Low threshold (0.1-0.2):** Catch more churners, more false positives
- **Medium threshold (0.3-0.5):** Balanced approach
- **High threshold (0.6+):** Only very certain cases, miss many churners

**Recommendation:** Use ~0.12 for XGBoost based on your data

## 📚 Files Generated

| File | Purpose |
|------|---------|
| `data/predictions_with_risk.csv` | All customers with risk scores |
| `results_3month/xgboost_churn_drivers.png` | Visual feature importance |
| `results_3month/segments/*.csv` | High-risk customers by segment |
| `results_3month/optimal_thresholds.json` | Optimal thresholds per model |

## 🚀 Next Steps

1. ✅ Run the complete workflow on your data
2. ✅ Review high-risk customer segments
3. ✅ Design interventions for each segment
4. ✅ Implement in CRM/marketing automation
5. ✅ Measure churn reduction over time

**Remember:** The goal isn't prediction accuracy - it's **reducing churn through actionable insights**!
