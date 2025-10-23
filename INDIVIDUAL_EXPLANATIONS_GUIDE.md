# Individual Prediction Explanations - Feature Breakdown Guide

## 📍 WHERE TO FIND FEATURE-LEVEL EXPLANATIONS

You have **THREE levels** of feature analysis:

### 1. GLOBAL IMPORTANCE (All customers)
**Tool:** `analyze_churn_drivers.py`
**Shows:** Which features matter MOST for churn overall

**Example:**
```bash
python analyze_churn_drivers.py --input data/features.csv \
    --model models_3month/xgboost.joblib --top-n 15
```

**Output:** Top 3 features driving churn across ALL customers:
- Self-Service Adoption (33%)
- Late Payments (24%)
- Days Since Contact (15%)

### 2. SEGMENT IMPORTANCE (Groups of customers)
**Tool:** `analyze_churn_drivers.py --segment`
**Shows:** Which features matter for SPECIFIC GROUPS

**Example:**
```bash
python analyze_churn_drivers.py --input data/features.csv \
    --model models_3month/xgboost.joblib --segment --export-segments
```

**Output:** Segments like "Payment Issues" (5 customers @ 94% risk)
**Action:** Flexible payment plans

### 3. INDIVIDUAL EXPLANATIONS (One customer) ← **YOU ASKED FOR THIS!**
**Tool:** `explain_predictions_simple.py`
**Shows:** WHY THIS SPECIFIC CUSTOMER is at risk

**Example:**
```bash
python explain_predictions_simple.py --input data/features.csv \
    --model models_3month/xgboost.joblib --customer ACC-0136
```

---

## 🔍 EXAMPLE: Individual Customer Explanation (ACC-0136)

```
Churn Probability: 99.9%
Prediction: ⚠️  WILL CHURN

Top 3 Risk Factors:

1. late_payments_180d_sum
   • Customer's value: 6.00 (⚠️ VERY HIGH)
   • Safe customers avg: 1.86
   • Churned customers avg: 2.19
   • Analysis: 3.3σ ABOVE safe customer average!
   • Conclusion: This customer looks like churned customers ⚠️

2. current_month_revenue_trend_180d
   • Customer's value: 0.76 (⚠️ VERY LOW)
   • Safe customers avg: 1.00
   • Churned customers avg: 0.96
   • Analysis: 4.7σ BELOW safe customer average!
   • Conclusion: Revenue declining significantly!

3. late_payments_90d_min
   • Customer's value: 1.00 (⚠️ VERY HIGH)
   • Safe customers avg: 0.02
   • Churned customers avg: 0.07
   • Analysis: 6.3σ ABOVE safe customer average!

DIAGNOSIS: Payment issues + revenue decline
ACTION: Financial hardship outreach, flexible payment plan
```

---

## 📖 HOW TO USE

### Explain Top N Highest-Risk Customers
```bash
python explain_predictions_simple.py \
    --input data/features.csv \
    --model models_3month/xgboost.joblib \
    --top-n 5
```

### Explain a Specific Customer
```bash
python explain_predictions_simple.py \
    --input data/features.csv \
    --model models_3month/xgboost.joblib \
    --customer ACC-0123
```

### For 3-Month Predictions
```bash
python explain_predictions_simple.py \
    --input data/features.csv \
    --model models_3month/xgboost.joblib \
    --top-n 3
```

---

## 💡 UNDERSTANDING THE OUTPUT

### Z-Score (σ = sigma)
- **0σ** = Average (normal)
- **±1σ** = Unusual (1 in 6 customers)
- **±2σ** = Very unusual (1 in 44 customers)
- **±3σ** = Extremely unusual (1 in 370 customers)

### Value Indicators
- **⚠️ VERY HIGH/LOW** = More than 2σ from average (red flag!)
- **↑ High / ↓ Low** = 1-2σ from average (concerning)
- **~ Normal** = Within 1σ of average (okay)

### "Closer to churned customers"
- Means this customer's value is more similar to churned customers than safe customers
- Strong indicator they will churn

---

## 🎯 BUSINESS USE CASE

### Workflow
1. Get high-risk list from `predict.py`
2. Explain top 10 highest-risk with `explain_predictions_simple.py`
3. For each customer, read the top 3 risk factors
4. Design **PERSONALIZED** intervention based on their specific issues

### Example: Personalized Interventions

**Customer ACC-0136:**
- **Issue #1:** 6 late payments (6.3σ above average)
- **Issue #2:** Revenue declining 24% (4.7σ below trend)
- **Issue #3:** Self-service adoption low

**→ Action:** Call immediately, offer payment plan, show product value (ROI), offer training

---

**Customer ACC-0265:**
- **Issue #1:** 5 late payments
- **Issue #2:** Low engagement (days since contact high)
- **Issue #3:** High support tickets

**→ Action:** Assign dedicated support rep, flexible billing, increase touchpoint frequency

---

## 📊 THREE-TIER ANALYSIS WORKFLOW

### Step 1: GLOBAL - What drives churn overall?
**→ Tool:** `analyze_churn_drivers.py`
**Result:** "Self-service adoption is #1 driver"
**Action:** Improve self-service tools company-wide

### Step 2: SEGMENT - Which groups need help?
**→ Tool:** `analyze_churn_drivers.py --segment`
**Result:** "23 customers with payment issues @ 94% risk"
**Action:** Launch payment assistance program

### Step 3: INDIVIDUAL - Why is THIS customer at risk?
**→ Tool:** `explain_predictions_simple.py --customer ACC-0136`
**Result:** "ACC-0136: 6 late payments + revenue down 24%"
**Action:** Personal call, custom payment plan, ROI demo

---

## 📁 FILES CREATED

| File | Purpose |
|------|---------|
| `explain_predictions_simple.py` | Individual customer explanations |
| `analyze_churn_drivers.py` | Global/segment importance |
| `optimize_threshold.py` | Find optimal prediction thresholds |
| `predict.py` | Generate risk scores |
| `model_trainer_3month.py` | Train 3-month prediction models |

---

## 🚀 QUICK START

```bash
# 1. Get high-risk customers
python predict.py --model models_3month/xgboost.joblib \
    --input data/features.csv \
    --threshold 0.117 \
    --output predictions.csv

# 2. Explain top 10 highest-risk
python explain_predictions_simple.py \
    --input data/features.csv \
    --model models_3month/xgboost.joblib \
    --top-n 10

# 3. For specific high-value customer
python explain_predictions_simple.py \
    --input data/features.csv \
    --model models_3month/xgboost.joblib \
    --customer ACC-0136

# 4. Design personalized retention campaign based on their specific issues
```

---

## 💡 KEY TAKEAWAYS

1. ✅ **Global analysis** shows what to fix company-wide
2. ✅ **Segment analysis** shows which groups need targeted programs
3. ✅ **Individual analysis** enables personalized outreach
4. ✅ Use Z-scores to identify **extreme** values (>2σ = red flag!)
5. ✅ Compare to "churned customer average" to see similarities
6. ✅ Top 3 risk factors = your talking points for retention call

**Bottom line:** Don't treat all at-risk customers the same. Use individual explanations to personalize your retention efforts!
