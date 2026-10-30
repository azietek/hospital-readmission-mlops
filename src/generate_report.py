import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def save_model_report(y_true, y_pred, y_prob, output_path="MODEL_REPORT.md"):

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc_score = roc_auc_score(y_true, y_prob)

    markdown_content = f"""# 📊 XGBoost Model Evaluation Report

## 1. Global Performance Metrics
* **ROC AUC:** `{auc_score:.4f}`
* **Accuracy:** `{report_dict['accuracy']:.4f}`
* **Macro F1-Score:** `{report_dict['macro avg']['f1-score']:.4f}`

---

## 2. Classification Metrics by Class
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (No Readmission)** | {report_dict['0']['precision']:.4f} | {report_dict['0']['recall']:.4f} | {report_dict['0']['f1-score']:.4f} | {report_dict['0']['support']} |
| **1 (Readmitted <30 Days)** | {report_dict['1']['precision']:.4f} | {report_dict['1']['recall']:.4f} | {report_dict['1']['f1-score']:.4f} | {report_dict['1']['support']} |

---

## 3. Confusion Matrix Breakdown
* **True Negatives (TN):** `{tn}`
* **False Positives (FP):** `{fp}`
* **False Negatives (FN):** `{fn}`
* **True Positives (TP):** `{tp}`
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"✅ Raport zapisany pomyślnie w `{output_path}`")
