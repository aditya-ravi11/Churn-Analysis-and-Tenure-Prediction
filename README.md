# Customer Churn Analysis and Tenure Prediction using Machine Learning

This project focuses on predicting customer churn using a combination of machine learning techniques, exploratory data analysis, feature engineering, and model interpretability tools. The solution is designed to provide business-ready insights for churn mitigation and customer retention strategies using real-world behavioral data.

---

## Project Goals

- Predict the likelihood of customer churn using advanced ML models.
- Estimate **expected tenure** of users using regression.
- Identify key behavioral indicators driving churn.
- Visualize churn cohorts, trends, and actionable insights.
- Provide a foundation for churn reduction & lifetime value prediction.

---

## Datasets Used

- **Training Set**: 499,238 rows
- **Testing Set**: 280,000+ rows
- Contains customer activity logs, engagement metrics, usage trends, and churn labels.

---

## Techniques & Methodology

### Feature Engineering:
- `expected_tenure_pred` (predicted via regression models)
- Rolling engagement metrics (e.g., `ema_30d`, `drop_trend`)
- Usage pattern aggregation (`zero_days`, `usage_cv`, `total_usage`)

### ML Models for Churn Classification:
- **RandomForestClassifier**
- **XGBoostClassifier**
- **LightGBMClassifier**
- **CatBoostClassifier**
- **LogisticRegression**
- **VotingClassifier (Ensemble)** – *Selected as final model*

### ML Model for Tenure Prediction:
- **RandomForestRegressor**
- **LightGBMRegressor**
- Final MAE: `~2.31`  
- Final R²: `0.939`

---

## Performance Summary

| Metric        | Value (on Test Data) |
|---------------|----------------------|
| Accuracy      | `0.81`               |
| F1 Score      | `0.82`               |
| AUC-ROC       | `0.880`              |
| Recall (Churn)| `0.87`               |
| Precision     | `0.55`               |

---

## Visual Insights & Analysis

- **Cohort Analysis**: Churn rates across `tenure` and `usage` segments.
- **Churn Heatmaps**: Visual segmentation of churn behavior.
- **Feature Importance**: Tree-based ranking of most influential features.
- **Lift Curve (Cumulative Gain)**: Evaluates model lift vs. random baseline.
- **Classification Reports**: Precision, Recall, F1 Score metrics.

---

## Business Insights

- Users with **low expected tenure** and **low usage** showed the **highest churn rates**.
- Targeted interventions in the `<30d` tenure cohort may yield best ROI.
- Potential integration with CLV models and retention campaigns.

---

## Future Work

- Add Customer Lifetime Value (CLV) Prediction
- Incorporate deep learning models for time-series trends
- Build fully interactive dashboards (Tableau or Streamlit)
- Deploy REST API for real-time churn inference

---

## Repository Contents

- `notebooks/`: Jupyter Notebooks for model training, evaluation
- `models/`: Saved model binaries (`.pkl` or `.joblib`)
- `visuals/`: Key images, graphs, cohort heatmaps
- `README.md`: Project overview

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Aditya Ravi**  
B.Tech in Artificial Intelligence & Data Science  
