# 📋 Project Summary: Hospital Readmission Prediction (MLOps)

## 1. Business & Clinical Objective
The goal of this project was to build a machine learning model to identify diabetic patients at high risk of 30-day hospital readmission. Key requirements included high reproducibility (PySpark), full transparency of decisions (SHAP), and a high patient capture rate (Recall).

## 2. Key Insights from Feature Engineering
* **Age Transformation Breakthrough:** Replacing artificial age bins with continuous midpoint values (`age_numeric`) allowed the model to naturally capture the non-linear increase in risk with age without excessive overfitting.
* **Discharge Aggregation:** Consolidating complex billing codes into clear categories (`Home`, `Transfer_Rehab`, `Home_Health`) removed noise and strengthened the diagnostic signal.
* **Cleaned Endpoint:** Removing deaths and hospice care protected the model from learning false correlations.

## 3. Model Interpretability & Feature Impact (SHAP Analysis)
Analysis of the SHAP summary and bar plots (`reports/shap_summary.png` and `reports/shap_bar.png`) confirms that the model’s decision-making aligns strongly with clinical intuition, driven by a clear hierarchy of risk factors:

* **`number_inpatient` (Top Predictor):** By far the most influential feature in the model. A high frequency of prior inpatient visits strongly increases the predicted probability of readmission.
* **`discharge_group_Home` (Protective Factor):** Discharging a patient directly to home significantly *decreases* the likelihood of readmission within 30 days.
* **`discharge_group_Transfer_Rehab / Institution` (Risk Factor):** Conversely, being transferred to an institution, specialized care, or a rehabilitation facility elevates the predicted readmission risk.
* **`total_prior_visits`:** Captures broader historical healthcare utilization, consistently pushing the risk score upward for frequent system users.
* **Secondary Clinical Indicators:** A robust tier of supportive features fine-tunes the model's predictions, including age (`age_numeric`), length of stay (`time_in_hospital`), total medications (`num_medications`), active diabetes medication status (`diabetesMed_Yes`), global utilization intensity (`total_activity`), and specific home-related encoding groups.

## 4. Summary of Results
The resulting model combines a stable **ROC-AUC (0.6661)** metric with high sensitivity to the minority class (**Recall 57.16%**). As a result, the system successfully alerts staff to over half of the patients who will actually return to the hospital, providing a solid foundation for deploying a Clinical Decision Support System (CDSS).