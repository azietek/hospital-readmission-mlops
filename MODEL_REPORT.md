# 📊 XGBoost Model Evaluation Report

## 1. Global Performance Metrics
* **ROC AUC:** `0.5668`
* **Accuracy:** `0.8854`
* **Macro F1-Score:** `0.4747`

---

## 2. Classification Metrics by Class
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (No Readmission)** | 0.8873 | 0.9976 | 0.9392 | 18054.0 |
| **1 (Readmitted <30 Days)** | 0.2143 | 0.0052 | 0.0102 | 2300.0 |

---

## 3. Confusion Matrix Breakdown
* **True Negatives (TN):** `18010`
* **False Positives (FP):** `44`
* **False Negatives (FN):** `2288`
* **True Positives (TP):** `12`
