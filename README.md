
# Churn Analysis & Tenure Prediction

An end-to-end machine learning pipeline to:
- Predict customer churn
- Estimate expected user tenure
- Generate actionable insights to drive retention strategies

---

## Dataset Overview

- **Train Set**: 624,048 rows × 33 columns  
- **Test Set**: 156,012 rows × 32 columns  
- Dataset consists of anonymized usage data across 90 days, engineered into 60-day summaries for clean feature learning.

---

## Project Pipeline

### 1. Data Preprocessing & Feature Engineering
- Used Day 1–60 only to avoid forward leakage.
- Engineered features: `ema_7d/14d/30d`, `std_7d/14d/30d`, `drop_trend`, `usage_slope`, `usage_cv`, `zero_days`, and more.
- Removed `last_active`, `churn`, `expected_tenure` from training features to prevent leakage.
- StandardScaler applied on numeric columns.

### 2. Expected Tenure Prediction
- Model: `RandomForestRegressor`
- **MAE**: 1.92 (train), **2.31 (test)**  
- **R²**: 0.956 (train), **0.939 (test)**

### 3. Churn Classification
- Best model: `VotingEnsemble`
- Trained with `expected_tenure_pred` as a feature (not target), avoiding actual tenure to prevent leakage.
- Models compared: `RandomForest`, `LightGBM`, `CatBoost`, `XGBoost`, `LogisticRegression`

---

## Model Results

### Churn Prediction – Test Set
| Metric       | Value      |
|--------------|------------|
| Accuracy     | 0.81       |
| Precision    | 0.56       |
| Recall       | **0.83**   |
| F1-Score     | 0.67       |
| AUC (ROC)    | **0.88**   |
| AUC (PR)     | 0.63       |

### Tenure Prediction – Test Set
| Metric       | Value   |
|--------------|---------|
| MAE          | 2.31    |
| R² Score     | 0.939   |

---

## Feature Importance (Top 5)
1. `drop_trend`
2. `usage_slope`
3. `zero_days`
4. `usage_cv`
5. `ema_14d`

> Full plot is available in the notebook.

---

## Churn Distribution

| Dataset     | Churn % |
|-------------|---------|
| Train Set   | 23.1%   |
| Test Set    | 22.9%   |

No SMOTE or class balancing used (yet). The model handles imbalance via high recall and feature discrimination.

---

## Visuals Included
- ROC and PR Curves (train/test)
- Feature Importance Plot
- Churn Distribution Bar Graphs

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/aditya-ravi11/Churn-Analysis-and-Tenure-Prediction.git
cd Churn-Analysis-and-Tenure-Prediction

# Create environment and install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook churn_analysis.ipynb
```

---

## Project Structure

```
Churn-Analysis-and-Tenure-Prediction/
│
├── churn_analysis.ipynb            # Main notebook
├── churn_prediction_model.pkl      # Final churn classification model
├── expected_tenure_model.pkl       # Final tenure prediction model
├── README.md                       # This file
└── .gitattributes                  # Git LFS tracking
```

---

## Next Steps / TODO
- [ ] Add SHAP plots for churn model interpretability
- [ ] Introduce class balancing (SMOTE / undersampling)
- [ ] Export results as CSV business reports
- [ ] Optional: Build a Streamlit dashboard for demo

---

## Key Insight

> Users with **<60 days tenure** and **low usage frequency** are the most likely to churn. Early-stage engagement and onboarding are mission-critical for retention.

## License
This project is licensed under the MIT License — you are free to use, modify, and distribute it with attribution.
See the LICENSE file for full details.

## Author
Aditya Ravi
B.Tech – Artificial Intelligence & Data Science
Mumbai, India
