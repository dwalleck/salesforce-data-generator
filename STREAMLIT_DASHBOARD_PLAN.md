# Streamlit Dashboard Implementation Plan

## Overview

This document outlines two implementation tiers for a churn prediction dashboard using Streamlit.

---

## 🚀 QUICK PROTOTYPE (30-45 minutes)

### Goal
Validate the dashboard approach with minimal features. Get something running quickly to see if Streamlit fits your needs.

### Features

#### 1. Data Loading
- File uploader for predictions CSV
- Or dropdown to select from existing prediction files
- Display basic data preview (first 10 rows)

#### 2. Overview Metrics (4 KPI cards)
- Total customers analyzed
- High-risk count (risk score > 0.7)
- Medium-risk count (0.4 - 0.7)
- Low-risk count (0.2 - 0.4)

#### 3. High-Risk Customer Table
- Show top 20 highest-risk customers
- Columns: account_id, churn_risk_score, risk_tier, prediction
- Sortable by risk score (default: descending)

#### 4. Risk Distribution Chart
- Simple bar chart showing count by risk tier
- Color-coded (Red = High, Orange = Medium, Yellow = Low, Green = Safe)

#### 5. Model Performance (if metrics.json available)
- Display accuracy, precision, recall, F1 score
- Simple confusion matrix display (text-based)

### File Structure

```
dashboard_prototype.py          # Main Streamlit app (~200 lines)
├── imports
├── load_data()                 # Load predictions CSV
├── display_overview_metrics()  # KPI cards
├── display_high_risk_table()   # Top 20 table
├── display_risk_chart()        # Bar chart
├── display_model_metrics()     # Performance metrics
└── main()                      # Streamlit app flow
```

### Dependencies to Add

```toml
# Add to pyproject.toml [project.dependencies]
streamlit = "^1.28.0"
plotly = "^5.18.0"  # For interactive charts
```

### Running the Prototype

```bash
uv sync
uv run streamlit run dashboard_prototype.py
```

### User Flow

```
1. User opens dashboard (localhost:8501)
2. Upload predictions.csv OR select from dropdown
3. See 4 KPI metrics at top
4. Scroll down to see high-risk customer table
5. View risk distribution bar chart
6. Check model performance metrics
```

### Estimated Lines of Code: ~200

---

## 🎯 INTERMEDIATE VERSION (4-6 hours)

### Goal
Production-ready dashboard with interactive filtering, multiple views, segmentation, and export capabilities.

### Features

#### 1. Enhanced Data Loading
- Upload predictions CSV
- Upload features CSV (for individual explanations)
- Upload trained model (.joblib) for live scoring
- Dropdown to select existing files
- Data validation and error handling
- Cache loaded data for performance

#### 2. Sidebar Navigation
- Radio buttons to switch between pages:
  - 📊 Overview
  - 🎯 High-Risk Customers
  - 👥 Customer Segments
  - 📈 Model Insights
  - 🔍 Individual Lookup
  - ⚙️ Settings

#### 3. Overview Page (Enhanced)
- **KPI Cards** (6 metrics)
  - Total customers
  - High/Medium/Low/Safe counts
  - Average risk score
  - Predicted churners (at current threshold)

- **Risk Distribution**
  - Interactive Plotly bar chart
  - Hover to see counts and percentages

- **Risk Score Histogram**
  - Distribution of risk scores (0-1)
  - Threshold line overlay

- **Model Comparison** (if both 1-month and 3-month available)
  - Side-by-side metrics
  - Highlight which is better

#### 4. High-Risk Customers Page
- **Filters** (sidebar)
  - Risk tier multiselect (High, Medium, Low)
  - Risk score range slider (0.0 - 1.0)
  - Search by account_id

- **Interactive Table**
  - All high-risk customers (risk > 0.2)
  - Sortable by any column
  - Columns: account_id, risk_score, risk_tier, top_risk_factor (if available)
  - Click row → navigate to individual lookup

- **Export Button**
  - Download filtered results as CSV
  - Include customer IDs and risk scores for CRM import

#### 5. Customer Segments Page
- **Segment Cards**
  - Payment Issues (count, avg risk, recommended action)
  - Support Issues (count, avg risk, recommended action)
  - Declining Engagement (count, avg risk, recommended action)
  - Revenue Decline (count, avg risk, recommended action)

- **Segment Selector**
  - Dropdown to choose segment
  - Show customers in that segment
  - Export segment to CSV

- **Recommended Actions**
  - Pre-defined retention strategies per segment
  - Customizable text boxes for notes

#### 6. Model Insights Page
- **Feature Importance Chart**
  - Interactive Plotly bar chart
  - Top 20 features
  - Hover to see exact importance scores

- **Confusion Matrix**
  - Plotly heatmap (if test data available)
  - Annotations with counts

- **ROC and PR Curves**
  - Side-by-side plots
  - Show AUC scores

- **Performance Metrics Table**
  - Accuracy, Precision, Recall, F1, ROC AUC, PR AUC
  - Color-coded (green = good, yellow = ok, red = poor)

#### 7. Individual Lookup Page
- **Search Bar**
  - Enter account_id
  - Auto-suggest from available customers

- **Customer Risk Card**
  - Large risk score display
  - Risk tier badge
  - Prediction (Will Churn / Safe)

- **Top 10 Risk Factors**
  - Table showing:
    - Feature name
    - Customer's value
    - Safe customer average
    - Churned customer average
    - Z-score
    - Risk contribution
  - Color-coded rows (red = very concerning)

- **Action Recommendations**
  - Based on top 3 risk factors
  - Pre-populated suggestions
  - Editable notes section

#### 8. Settings Page
- **Threshold Adjustment**
  - Slider (0.0 - 1.0)
  - Show resulting precision/recall
  - Update predictions in real-time

- **Model Selection**
  - Dropdown: 1-month vs 3-month
  - Display model info (training date, performance)

- **Display Options**
  - Number of rows in tables (10, 20, 50, 100)
  - Chart color scheme
  - Decimal precision

### File Structure

```
streamlit_dashboard.py          # Main app entry point (~100 lines)
├── pages/
│   ├── 1_📊_Overview.py        # Overview page (~150 lines)
│   ├── 2_🎯_High_Risk.py       # High-risk customers (~200 lines)
│   ├── 3_👥_Segments.py        # Segmentation (~150 lines)
│   ├── 4_📈_Model_Insights.py  # Model performance (~150 lines)
│   └── 5_🔍_Individual.py      # Individual lookup (~200 lines)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # Load predictions, features, models (~100 lines)
│   ├── metrics.py              # Calculate metrics (~80 lines)
│   ├── segmentation.py         # Customer segmentation logic (~120 lines)
│   ├── explanations.py         # Individual explanations (~100 lines)
│   └── visualizations.py       # Reusable chart functions (~150 lines)
│
├── config/
│   └── dashboard_config.py     # Configuration (colors, thresholds, etc.) (~50 lines)
│
└── README_DASHBOARD.md         # Usage instructions
```

### Enhanced Dependencies

```toml
# Add to pyproject.toml [project.dependencies]
streamlit = "^1.28.0"
plotly = "^5.18.0"
streamlit-aggrid = "^0.3.4"  # For advanced tables
```

### Running the Intermediate Dashboard

```bash
uv sync
uv run streamlit run streamlit_dashboard.py
```

### User Flows

#### Flow 1: Identify High-Risk Customers for Outreach
```
1. Navigate to "High-Risk Customers" page
2. Filter to "High Risk" tier only
3. Sort by risk score (descending)
4. Review top 20 customers
5. Click "Export to CSV"
6. Import into CRM for outreach campaign
```

#### Flow 2: Understand Individual Customer Risk
```
1. Navigate to "Individual Lookup" page
2. Search for customer (e.g., ACC-0136)
3. See risk score (e.g., 99.9%)
4. Review top 10 risk factors:
   - 6 late payments (6.3σ above average)
   - Revenue declining 24%
5. Read recommended actions:
   - Call immediately
   - Offer payment plan
   - Show ROI demo
6. Add notes and export customer profile
```

#### Flow 3: Build Segment-Specific Retention Campaign
```
1. Navigate to "Customer Segments" page
2. Select "Payment Issues" segment
3. See 23 customers @ 94% average risk
4. Review recommended action: "Flexible payment plans"
5. Export segment to CSV
6. Design targeted 90-day retention campaign
```

#### Flow 4: Optimize Prediction Threshold
```
1. Navigate to "Settings" page
2. Adjust threshold slider (0.5 → 0.117)
3. See precision/recall tradeoff update
4. Note: Lower threshold = catch more churners, more false alarms
5. Save optimal threshold
6. Return to "High-Risk Customers" with new predictions
```

### Key Technical Decisions

#### Caching Strategy
```python
@st.cache_data
def load_predictions(file_path):
    """Cache predictions to avoid reloading on every interaction."""
    return pd.read_csv(file_path)

@st.cache_resource
def load_model(model_path):
    """Cache model in memory (persist across reruns)."""
    return joblib.load(model_path)
```

#### Session State Management
```python
# Store user selections across page navigation
if 'selected_customer' not in st.session_state:
    st.session_state.selected_customer = None

if 'risk_threshold' not in st.session_state:
    st.session_state.risk_threshold = 0.5
```

#### Multi-Page Architecture
Streamlit supports two approaches:
1. **Single file with tabs** (Quick Prototype approach)
2. **Multi-page with pages/ directory** (Intermediate approach - recommended)

We'll use approach #2 for better organization.

### Estimated Lines of Code: ~1,200

---

## 🚀 ADVANCED VERSION (8-12 hours)

### Goal
Enterprise-grade dashboard with advanced analytics, automated workflows, time-series tracking, custom segmentation, and model retraining capabilities.

### Features

#### All Intermediate Features, Plus:

#### 1. Enhanced Individual Customer Drill-Down
**Upgrade from Intermediate's Individual Lookup page**

- **Clickable High-Risk Table**
  - Click any row in High-Risk Customers table → navigate to drill-down
  - Deep-link support (bookmark specific customer pages)

- **Comprehensive Customer Profile**
  - Risk score trend chart (last 6 months)
  - Historical feature values (show how late_payments changed over time)
  - Churn probability trajectory
  - Comparison to similar customers (same segment)

- **Visual Feature Breakdown**
  - Waterfall chart showing how each feature contributes to risk score
  - Radar chart comparing customer to safe/churned averages
  - Interactive feature explorer (click feature → see distribution)

- **Action History**
  - Log of retention actions taken (editable notes)
  - Outcome tracking (did they churn? did intervention work?)
  - Campaign assignment (which retention campaign is this customer in?)

- **Smart Recommendations Engine**
  - ML-generated recommendations based on similar customers who were saved
  - Success rate of each recommendation (based on historical data)
  - Estimated intervention cost vs customer lifetime value

#### 2. Advanced Threshold Optimization
**Upgrade from Intermediate's Settings page**

- **Interactive Precision-Recall Curve**
  - Draggable threshold slider on actual PR curve
  - Real-time update of metrics as you drag
  - Show number of customers affected at each threshold

- **Cost-Benefit Calculator**
  - Input: Cost of retention effort per customer
  - Input: Average customer lifetime value
  - Output: ROI at different thresholds
  - Recommendation: Optimal threshold for max profit

- **Multi-Objective Optimization**
  - Maximize F1 score
  - Maximize recall (catch all churners)
  - Minimize false positives (avoid alert fatigue)
  - Custom: Set minimum precision constraint

- **Threshold Comparison Table**
  - Show 5-10 different thresholds side-by-side
  - Metrics, costs, churners caught for each
  - Highlight recommended threshold

- **A/B Testing Setup**
  - Split customers into test groups
  - Apply different thresholds to each group
  - Track which performs better over time

#### 3. Time-Series Risk Tracking
**New page: 📈 Risk Trends**

- **Customer Risk Trajectory**
  - Select customer → see risk score over last 6-12 months
  - Annotate with intervention dates
  - Show feature changes that drove risk increases/decreases

- **Cohort Analysis**
  - Track risk for customer cohorts (e.g., "Joined Q1 2024")
  - Compare cohorts (early churners vs late churners)
  - Identify at-risk cohorts

- **Portfolio Risk Dashboard**
  - Overall portfolio churn risk (weighted by customer value)
  - Trend: Is risk increasing or decreasing?
  - Forecasted churn for next quarter
  - Alert if sudden risk spike detected

- **Feature Drift Monitoring**
  - Track how feature distributions change over time
  - Alert if data drift detected (model may need retraining)
  - Compare current month to training data

- **Intervention Impact Analysis**
  - Show average risk change after retention campaigns
  - Which interventions are most effective?
  - ROI tracking by campaign type

#### 4. Custom Segment Builder
**Upgrade from Intermediate's Segments page**

- **Visual Query Builder**
  - Drag-and-drop interface for building segments
  - Filter by any combination of features:
    - late_payments > 3
    - AND self_service_percentage < 0.5
    - AND risk_score > 0.7
  - Real-time customer count as you build

- **Saved Segments**
  - Save custom segments with names
  - Examples:
    - "High-value at-risk" (revenue > 10k, risk > 0.7)
    - "Payment issues + support burden"
    - "Recently declined engagement"
  - Share segments with team

- **Segment Comparison**
  - Compare 2-3 segments side-by-side
  - Which segment has higher risk?
  - Which features differ the most?
  - Which segment has better retention rate?

- **Automated Segment Discovery**
  - ML-based clustering to find natural segments
  - Suggest segments based on churn drivers
  - Identify "hidden" at-risk groups

- **Segment Performance Tracking**
  - For each segment, track:
    - How many customers
    - Average risk score
    - Churn rate over time
    - Retention campaign effectiveness
  - Alert if segment risk increases

#### 5. Automated Retention Campaign Builder
**New page: 🎯 Campaign Builder**

- **Campaign Templates**
  - Pre-built templates:
    - Payment assistance program
    - Support escalation
    - Product training campaign
    - Win-back offer
    - Executive outreach
  - Customizable templates

- **Campaign Wizard**
  - Step 1: Select segment or customers
  - Step 2: Choose campaign type
  - Step 3: Set timeline (e.g., 90-day outreach)
  - Step 4: Assign actions (call, email, discount)
  - Step 5: Set success criteria

- **Action Assignment**
  - Assign customers to team members
  - Generate task list for each rep:
    - "Call ACC-0136 by Friday"
    - "Email payment plan to ACC-0265"
  - Export to CSV for CRM import

- **Email & Call Scripts**
  - Auto-generate personalized scripts:
    - "Hi [Name], we noticed you've had [6] late payments..."
    - "Based on your usage, we recommend..."
  - Talking points based on top risk factors

- **Campaign Tracking**
  - Track campaign progress:
    - Customers contacted: 23/50
    - Responded: 8/23
    - Risk decreased: 5/8
  - Measure campaign ROI

- **Automated Exports**
  - Schedule daily/weekly exports to CRM
  - Email high-risk list to retention team
  - Slack/Teams notifications for urgent cases

- **Multi-Touch Campaign Builder**
  - Day 1: Send email
  - Day 3: Follow-up call if no response
  - Day 7: Offer discount
  - Day 14: Escalate to manager
  - Auto-generate timeline

#### 6. Model Retraining Interface
**New page: 🔧 Model Training**

- **Data Upload & Validation**
  - Upload new customer data (CSV)
  - Validate schema matches expected format
  - Show data quality report:
    - Missing values
    - Outliers
    - Feature distributions
  - Compare to existing training data (check for drift)

- **Feature Engineering**
  - Run feature engineering pipeline on new data
  - Show generated features preview
  - Verify feature counts match expected

- **Train/Test Split Configuration**
  - Choose split strategy:
    - Time-based (recommended)
    - Random
    - Custom
  - Set train/val/test percentages
  - Preview split results

- **Model Selection**
  - Choose models to train:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - All of the above
  - Enable hyperparameter tuning (checkbox)
  - Set cross-validation folds

- **Training Progress**
  - Live training progress bar
  - Show current step:
    - "Handling class imbalance with SMOTE..."
    - "Training XGBoost... (2/3 models)"
    - "Evaluating on test set..."
  - Estimated time remaining

- **Model Comparison**
  - Side-by-side comparison table:
    - Old model vs new model
    - Metrics: Accuracy, Precision, Recall, F1, ROC AUC
    - Color-coded improvements (green) / regressions (red)
  - Feature importance comparison
  - Confusion matrix comparison

- **Model Approval Workflow**
  - Review new model performance
  - If better: "Promote to Production" button
  - If worse: "Reject and Keep Old Model"
  - If uncertain: "Deploy to A/B Test"

- **Model Versioning**
  - Track model history:
    - v1.0: Trained 2024-01-15, Recall: 27.8%
    - v2.0: Trained 2024-02-20, Recall: 42.5% ✅ Current
    - v3.0: Trained 2024-03-18, Recall: 38.2% (rejected)
  - Rollback to previous version
  - Export model (.joblib) for offline use

- **Automated Retraining Schedule**
  - Set retraining cadence:
    - Monthly
    - Quarterly
    - When data drift detected
  - Email notification when training completes
  - Auto-promote if new model is >5% better

#### 7. Additional Advanced Features

**Real-Time Scoring**
- Upload single customer data → get instant risk score
- Batch scoring: Upload 100 customers → score all at once
- API endpoint generation (for CRM integration)

**What-If Analysis**
- "What if this customer had 0 late payments?"
- Adjust feature values → see risk score change
- Identify which changes would reduce risk the most

**Explainable AI Enhancements**
- SHAP force plots (if SHAP compatibility fixed)
- LIME explanations as alternative
- Natural language explanations:
  - "This customer is at risk because their late payments are 6.3σ above average"

**Alert System**
- Set custom alerts:
  - Email me if any customer's risk increases >20% in a month
  - Notify team if high-value customer (revenue > 50k) becomes high-risk
  - Daily digest of new high-risk customers
- Alert history and management

**Data Quality Dashboard**
- Missing value tracking
- Feature correlation heatmap
- Outlier detection
- Data freshness indicators

### File Structure

```
streamlit_dashboard_advanced.py     # Main app entry point (~150 lines)
├── pages/
│   ├── 1_📊_Overview.py            # Overview page (~200 lines)
│   ├── 2_🎯_High_Risk.py           # High-risk customers with drill-down (~300 lines)
│   ├── 3_👥_Segments.py            # Custom segment builder (~250 lines)
│   ├── 4_📈_Model_Insights.py      # Model performance (~200 lines)
│   ├── 5_🔍_Individual.py          # Enhanced customer drill-down (~350 lines)
│   ├── 6_📈_Risk_Trends.py         # Time-series tracking (~250 lines)
│   ├── 7_🎯_Campaign_Builder.py    # Retention campaigns (~300 lines)
│   ├── 8_🔧_Model_Training.py      # Retraining interface (~400 lines)
│   └── 9_⚙️_Settings.py            # Advanced settings (~200 lines)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py              # Load predictions, features, models (~150 lines)
│   ├── metrics.py                  # Calculate metrics (~120 lines)
│   ├── segmentation.py             # Customer segmentation + custom builder (~200 lines)
│   ├── explanations.py             # Individual explanations (~150 lines)
│   ├── visualizations.py           # Reusable chart functions (~250 lines)
│   ├── time_series.py              # Risk tracking over time (~180 lines)
│   ├── campaigns.py                # Campaign generation logic (~200 lines)
│   ├── model_training.py           # Retraining pipeline wrapper (~250 lines)
│   ├── alerts.py                   # Alert system (~120 lines)
│   └── api.py                      # API endpoint generation (~100 lines)
│
├── components/
│   ├── __init__.py
│   ├── customer_card.py            # Reusable customer profile card (~80 lines)
│   ├── risk_gauge.py               # Visual risk gauge component (~60 lines)
│   ├── comparison_table.py         # Side-by-side comparison widget (~100 lines)
│   ├── query_builder.py            # Visual segment builder (~150 lines)
│   └── threshold_optimizer.py      # Interactive threshold widget (~120 lines)
│
├── config/
│   ├── dashboard_config.py         # Configuration (~80 lines)
│   ├── campaign_templates.py       # Retention campaign templates (~100 lines)
│   └── alert_rules.py              # Alert rule definitions (~60 lines)
│
├── database/
│   ├── __init__.py
│   ├── models.py                   # SQLAlchemy models (if using DB) (~150 lines)
│   └── crud.py                     # CRUD operations (~100 lines)
│
└── README_DASHBOARD.md             # Comprehensive usage guide
```

### Enhanced Dependencies

```toml
# Add to pyproject.toml [project.dependencies]
streamlit = "^1.28.0"
plotly = "^5.18.0"
streamlit-aggrid = "^0.3.4"         # Advanced tables
pandas = "^2.1.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
xgboost = "^2.0.0"
joblib = "^1.3.0"

# Optional but recommended
sqlalchemy = "^2.0.0"               # For database tracking
apscheduler = "^3.10.0"             # For scheduled retraining
smtplib = "*"                       # For email alerts (built-in)
requests = "^2.31.0"                # For webhook alerts
streamlit-extras = "^0.3.0"         # Additional UI components
```

### Running the Advanced Dashboard

```bash
uv sync
uv run streamlit run streamlit_dashboard_advanced.py
```

Dashboard will open at `http://localhost:8501` with 9 pages.

### User Flows

#### Flow 1: Deep-Dive Customer Investigation
```
1. Navigate to "High-Risk Customers" page
2. See ACC-0136 at top with 99.9% risk
3. Click on ACC-0136 row → navigate to drill-down page
4. See comprehensive profile:
   - Risk trend: Steady at 30% for 4 months, spiked to 99.9% last month
   - Trigger: Late payments jumped from 1 to 6
   - Revenue declined 24% over 6 months
5. View waterfall chart: late_payments contributed +45% to risk score
6. Compare to similar customers: 3 similar customers churned after same pattern
7. See recommended actions:
   - Call within 48 hours (success rate: 67% when done early)
   - Offer payment plan (saved 8/12 similar customers)
   - Estimated intervention cost: $200, Customer LTV: $15,000
8. Add note: "Called on 2024-03-20, customer accepted payment plan"
9. Assign to "Payment Assistance Campaign"
10. Set follow-up reminder for 30 days
```

#### Flow 2: Build Custom High-Value Retention Campaign
```
1. Navigate to "Segments" page
2. Click "Create Custom Segment"
3. Use visual query builder:
   - current_month_revenue_180d_avg > 10000
   - AND churn_risk_score > 0.7
   - AND has_account_manager = False
4. See 8 customers match criteria
5. Save segment as "High-Value Unmanaged At-Risk"
6. Navigate to "Campaign Builder"
7. Select segment: "High-Value Unmanaged At-Risk"
8. Choose template: "Executive Outreach"
9. Configure campaign:
   - Day 1: Assign account manager
   - Day 3: Executive call
   - Day 7: Send ROI case study
   - Day 14: Offer discount or premium support
10. Generate email scripts (personalized for each customer)
11. Assign customers to team members:
    - Sarah: ACC-0136, ACC-0265
    - John: ACC-0478, ACC-0612
12. Export action list to CSV
13. Import into CRM
14. Track campaign progress over 90 days
15. Measure: 6/8 customers reduced risk below 0.5
```

#### Flow 3: Monthly Model Retraining
```
1. Navigate to "Model Training" page
2. Upload new customer data (march_2024_customers.csv)
3. See validation report:
   - ✅ Schema matches expected format
   - ✅ 11,533 records loaded
   - ⚠️  Warning: Average late_payments increased 15% (data drift detected)
4. Click "Run Feature Engineering"
5. Wait 30 seconds → see 107 features generated
6. Configure training:
   - Select: XGBoost only (fastest)
   - Enable hyperparameter tuning: Yes
   - Cross-validation folds: 3
7. Click "Train Model"
8. Watch progress:
   - Creating 3-month labels... ✓
   - Splitting data... ✓
   - Handling class imbalance... ✓
   - Training XGBoost... 67% complete
9. Training complete! See comparison:
   - Old model (v2.0): Recall 42.5%, Precision 88.3%
   - New model (v3.0): Recall 48.2%, Precision 86.1% ⬆️ Better recall!
10. Review feature importance: late_payments_180d_sum still #1
11. Check confusion matrix: Catching 94 churners vs 83 before
12. Click "Promote to Production"
13. New model saved as v3.0 (active)
14. Download model file for backup
15. Set automated retraining: Monthly on 1st
```

#### Flow 4: Optimize Threshold for Maximum ROI
```
1. Navigate to "Settings" page
2. Scroll to "Advanced Threshold Optimization"
3. Enter business parameters:
   - Cost per retention effort: $150
   - Average customer lifetime value: $12,000
   - Success rate of retention: 40%
4. See interactive PR curve with threshold slider
5. Drag slider from 0.5 to 0.117:
   - Churners caught: 13 → 83 (+70)
   - False positives: 0 → 11 (+11)
   - Total retention cost: $1,950 → $14,100
   - Expected saves: 5 customers → 33 customers
   - Expected revenue saved: $60k → $396k
   - Net ROI: $58k → $382k ⬆️ 6.5x better!
6. See recommendation: "Use threshold 0.117 for max profit"
7. Click "Apply Threshold"
8. All predictions update across dashboard
9. Export new high-risk list with 94 customers (vs 13 before)
```

#### Flow 5: Track Retention Campaign Effectiveness
```
1. Navigate to "Risk Trends" page
2. Select "Intervention Impact Analysis"
3. See campaign history:
   - Payment Assistance (Jan 2024): 23 customers
     - Average risk before: 94%
     - Average risk after (60 days): 38% ⬇️ 56 percentage points!
     - Churn prevented: 14/23 (61% save rate)
     - ROI: $201,000 (14 customers @ $15k LTV - $3,450 cost)
   - Support Escalation (Feb 2024): 15 customers
     - Average risk before: 87%
     - Average risk after (60 days): 82% ⬇️ Only 5 points
     - Churn prevented: 2/15 (13% save rate)
     - ROI: -$600 (not effective)
4. Insight: Payment assistance works, support escalation doesn't
5. Action: Discontinue support escalation campaign, expand payment assistance
6. Create new campaign based on learning
```

### Key Technical Decisions

#### Database Integration (Optional but Recommended)
```python
# Store campaign history, customer notes, model versions
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Campaign(Base):
    __tablename__ = 'campaigns'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    segment = Column(String)
    created_at = Column(DateTime)
    customers = Column(String)  # JSON list
    status = Column(String)  # active, completed, cancelled

class CustomerNote(Base):
    __tablename__ = 'customer_notes'
    id = Column(Integer, primary_key=True)
    account_id = Column(String)
    note = Column(String)
    created_at = Column(DateTime)
    created_by = Column(String)

# In Streamlit app
engine = create_engine('sqlite:///dashboard.db')
Base.metadata.create_all(engine)
```

#### Background Training Jobs
```python
from apscheduler.schedulers.background import BackgroundScheduler

def retrain_model():
    """Background job for automated retraining."""
    # Run feature engineering
    # Train model
    # Evaluate
    # If better, promote to production
    # Send email notification
    pass

scheduler = BackgroundScheduler()
scheduler.add_job(retrain_model, 'cron', day=1, hour=2)  # 2 AM on 1st of month
scheduler.start()
```

#### API Endpoint Generation
```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/predict")
def predict_churn(customer_data: dict):
    """API endpoint for real-time scoring."""
    model = load_model('models_3month/xgboost.joblib')
    features = preprocess(customer_data)
    risk_score = model.predict_proba(features)[0, 1]
    return {"risk_score": risk_score, "risk_tier": assign_risk_tier(risk_score)}

# In Streamlit dashboard: Button to "Generate API Endpoint"
# Outputs: curl command, Python code snippet, JavaScript example
```

#### State Persistence
```python
# Save user preferences, custom segments, alert rules
import json

def save_user_config(config_dict):
    with open('.streamlit/user_config.json', 'w') as f:
        json.dump(config_dict, f)

def load_user_config():
    try:
        with open('.streamlit/user_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Store: custom segments, alert rules, display preferences
```

### Estimated Lines of Code: ~4,200

**Breakdown:**
- Main app: 150 lines
- Pages (9): ~2,450 lines
- Utils (10 files): ~1,520 lines
- Components (5): ~510 lines
- Config (3): ~240 lines
- Database (2): ~250 lines
- Total: **~4,200 lines**

### Development Timeline (8-12 hours)

**Hour 1-2: Foundation**
- Set up multi-page structure
- Implement database models
- Create reusable components (customer card, risk gauge)
- Configure caching and session state

**Hour 3-4: Enhanced Individual Drill-Down**
- Clickable table navigation
- Customer profile with risk trend
- Waterfall charts and radar plots
- Action history logging

**Hour 5-6: Custom Segment Builder + Time-Series**
- Visual query builder
- Saved segments
- Risk trajectory charts
- Cohort analysis
- Intervention impact tracking

**Hour 7-8: Campaign Builder**
- Campaign templates
- Wizard flow
- Script generation
- Export functionality
- Multi-touch timeline builder

**Hour 9-10: Model Retraining Interface**
- Data upload and validation
- Feature engineering integration
- Training progress UI
- Model comparison
- Version management

**Hour 11-12: Advanced Threshold + Polish**
- Interactive PR curve with ROI calculator
- Cost-benefit analysis
- Alert system setup
- Testing and bug fixes
- Documentation

---

## 📊 Feature Comparison

| Feature | Quick Prototype | Intermediate | Advanced |
|---------|----------------|--------------|----------|
| **Data Upload** | ✅ Basic | ✅ Advanced (validation, caching) | ✅ + Real-time validation & drift detection |
| **KPI Metrics** | ✅ 4 cards | ✅ 6 cards + comparison | ✅ + Portfolio risk dashboard |
| **High-Risk Table** | ✅ Top 20 | ✅ All, filterable, sortable | ✅ + Clickable drill-down |
| **Risk Chart** | ✅ Basic bar | ✅ Interactive Plotly | ✅ + Time-series trends |
| **Model Metrics** | ✅ Text display | ✅ Full visual dashboard | ✅ + Version comparison |
| **Filtering** | ❌ | ✅ Multi-dimensional | ✅ + Visual query builder |
| **Export** | ❌ | ✅ CSV download | ✅ + Scheduled exports, CRM integration |
| **Segmentation** | ❌ | ✅ Full segment analysis | ✅ + Custom segments, ML clustering |
| **Individual Lookup** | ❌ | ✅ Detailed explanations | ✅ + Risk trajectory, what-if analysis |
| **Threshold Tuning** | ❌ | ✅ Interactive slider | ✅ + ROI calculator, A/B testing |
| **Multi-page Nav** | ❌ | ✅ 5 pages | ✅ 9 pages |
| **Model Comparison** | ❌ | ✅ 1-month vs 3-month | ✅ + Version history, rollback |
| **Time-Series Tracking** | ❌ | ❌ | ✅ Risk trends, cohort analysis |
| **Campaign Builder** | ❌ | ❌ | ✅ Templates, wizard, ROI tracking |
| **Model Retraining** | ❌ | ❌ | ✅ Full interface with automation |
| **Database Integration** | ❌ | ❌ | ✅ SQLite/PostgreSQL support |
| **Alert System** | ❌ | ❌ | ✅ Email/Slack notifications |
| **API Generation** | ❌ | ❌ | ✅ FastAPI endpoint creation |
| **What-If Analysis** | ❌ | ❌ | ✅ Feature manipulation |
| **Action History** | ❌ | ❌ | ✅ Campaign tracking & outcomes |

---

## 🚦 Decision Guide

### Choose Quick Prototype if:
- You want to validate the concept quickly
- You're not sure if Streamlit is right for you
- You only need basic risk score viewing
- Time is very limited (< 1 hour)
- This is a proof-of-concept or demo

### Choose Intermediate if:
- You want a production-ready tool
- Your team will use this regularly for retention planning
- You need filtering, segmentation, and export capabilities
- You want to justify retention ROI to stakeholders
- You're willing to invest 4-6 hours for a polished result
- You have basic churn prediction needs

### Choose Advanced if:
- You manage a large customer portfolio (1000+ customers)
- You need to track retention campaigns and measure ROI
- You want to retrain models regularly without code
- You need time-series risk tracking and cohort analysis
- You want to build custom retention campaigns with templates
- You need database persistence for campaign history
- You want alert notifications for high-risk customers
- You plan to integrate with CRM systems via API
- You have dedicated retention team that will use this daily
- You're willing to invest 8-12 hours for enterprise features
- Budget allows for more sophisticated tooling

---

## 🎯 Recommended Path

### Path 1: Incremental (Recommended for Most)

1. **Day 1 (30-45 min)**: Build Quick Prototype
   - Validates approach
   - Shows stakeholders what's possible
   - Gets early feedback
   - Decision point: Does Streamlit fit your needs?

2. **Day 2 (4-6 hours)**: Build Intermediate Version
   - Incorporate feedback from prototype
   - Add production features
   - Deploy for team use
   - Decision point: Do you need advanced features?

3. **Week 2 (8-12 hours)**: Upgrade to Advanced (Optional)
   - Add after Intermediate proves valuable
   - Build based on actual usage patterns
   - Add only the advanced features you need

### Path 2: Direct to Intermediate (Most Common)

Skip prototype if:
- You're confident Streamlit is right
- You've seen similar dashboards before
- You need production features from day 1

Build Intermediate (4-6 hours) → Deploy → Evaluate → Consider Advanced later

### Path 3: All-In Advanced (Enterprise)

Go straight to Advanced if:
- You have clear requirements for all advanced features
- You have a dedicated retention team waiting to use it
- You need campaign tracking and retraining from day 1
- Budget and timeline support 8-12 hour investment

Build Advanced (8-12 hours) → Deploy → Iterate based on feedback

### Path 4: Modular Advanced (Recommended for Enterprise)

Build Advanced in phases:

**Phase 1 (Week 1)**: Core Advanced Features (4-6 hours)
- Enhanced individual drill-down
- Time-series risk tracking
- Custom segment builder

**Phase 2 (Week 2)**: Campaign & Automation (3-4 hours)
- Campaign builder with templates
- Automated exports
- Alert system

**Phase 3 (Week 3)**: Model Management (2-3 hours)
- Model retraining interface
- Version management
- ROI threshold optimization

**Phase 4 (Week 4)**: Integration & Polish (2-3 hours)
- Database integration
- API generation
- What-if analysis
- Testing and documentation

**Total**: 11-16 hours spread over 4 weeks (more manageable)

---

## 📦 Installation & Setup

### Prerequisites
```bash
# Ensure uv project is set up
uv sync
```

### Add Dependencies
```bash
# Quick Prototype
uv add streamlit plotly

# Intermediate (additional)
uv add streamlit-aggrid

# Advanced (additional)
uv add sqlalchemy apscheduler requests streamlit-extras
```

### Launch Dashboard
```bash
# Quick Prototype
uv run streamlit run dashboard_prototype.py

# Intermediate
uv run streamlit run streamlit_dashboard.py

# Advanced
uv run streamlit run streamlit_dashboard_advanced.py
```

Dashboard will open at `http://localhost:8501`

---

## 🔄 Future Enhancements (Beyond Advanced)

If the Advanced version is successful, consider these next-level features:

1. **Authentication & Multi-User**:
   - User login with role-based access (Admin, Manager, Analyst)
   - Team collaboration features (shared segments, campaigns)
   - Audit logs (who did what, when)

2. **Live Database Integration**:
   - Connect directly to customer database (PostgreSQL, MySQL, Snowflake)
   - Real-time data sync (no CSV uploads needed)
   - Automatic daily predictions

3. **Advanced A/B Testing**:
   - Split customers into test/control groups
   - Track campaign effectiveness over time
   - Statistical significance testing
   - Multi-armed bandit optimization

4. **Mobile Responsive Dashboard**:
   - Optimize for tablet/phone access
   - Push notifications on mobile
   - Quick actions (approve campaign, add note)

5. **CRM Integration**:
   - Salesforce connector
   - HubSpot integration
   - Automatic sync of risk scores to CRM
   - Two-way sync (notes, actions)

6. **Advanced ML Features**:
   - Customer lifetime value prediction
   - Next best action recommendation (reinforcement learning)
   - Churn reason prediction (why they'll churn)
   - Automated feature engineering

7. **Executive Reporting**:
   - PDF report generation
   - Email scheduled reports (weekly churn summary)
   - PowerPoint export for board meetings
   - Custom KPI tracking

8. **Multi-Model Ensemble**:
   - Combine multiple models for better predictions
   - Voting classifier
   - Stacking models
   - Confidence intervals

**Note**: Many features previously listed here (database integration, alerts, retraining, API) are now included in the Advanced version!

---

## 📝 Next Steps

### To Proceed:

1. **Decide on Version**:
   - [ ] Quick Prototype (30-45 min) - Validate concept
   - [ ] Intermediate (4-6 hours) - Production ready
   - [ ] Advanced (8-12 hours) - Enterprise features
   - [ ] Path 4: Modular Advanced (11-16 hours over 4 weeks)

2. **Confirm Data Files Are Ready**:
   - `predictions.csv` (from predict.py)
   - `data/features.csv` (for individual explanations)
   - `models_3month/xgboost.joblib` (trained model)
   - `results_3month/metrics.json` (model performance)
   - `results_3month/feature_importance/` (charts)

3. **Install Dependencies**:
   ```bash
   # For Quick Prototype
   uv add streamlit plotly

   # For Intermediate (add to above)
   uv add streamlit-aggrid

   # For Advanced (add to above)
   uv add sqlalchemy apscheduler requests streamlit-extras
   ```

4. **Start Building**:
   - Create dashboard file (or let me build it for you)
   - Follow the implementation plan for your chosen version
   - Test with your actual data
   - Iterate based on feedback

### Implementation Options:

**Option A**: I build it for you
- You specify which version (Prototype, Intermediate, or Advanced)
- I create all necessary files
- You run and test
- We iterate based on your feedback

**Option B**: You build it yourself
- Use this plan as a guide
- Start with file structure
- Build page by page
- Ask me for help on specific sections

**Option C**: Collaborative
- I build core structure and first few pages
- You add additional pages based on the plan
- I help debug and add features as needed

### 🚀 Ready to Start?

**Which option would you like?**
- [ ] Build Quick Prototype (I'll create it now, 30-45 min)
- [ ] Build Intermediate Version (I'll create it, 4-6 hours)
- [ ] Build Advanced Version (I'll create it, 8-12 hours)
- [ ] Build Advanced - Phase 1 Only (Core features, 4-6 hours)
- [ ] Show me the code structure first (explain, don't build yet)

**If you choose to build, I'll:**
1. Create all necessary files
2. Add proper caching and session state
3. Include sample code and comments
4. Test with your data files
5. Provide usage documentation

Let me know which path you'd like to take!
