import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def generate_shap_values(model, X_test):
    """Generates SHAP values for the given model and test data."""
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    return shap_values

def save_shap_plots(shap_values, X_sample, output_dir="reports"):
    """
    Generates and saves SHAP summary plots for model interpretability:
    1. Summary Plot (beeswarm)
    2. Bar Plot (feature importance)
    """
    os.makedirs(output_dir, exist_ok=True)
    max_features = 10

    # --- 1. SUMMARY PLOT (Beeswarm) ---
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_sample, max_display=max_features, show=False)
    plt.title("SHAP Summary Plot", fontsize=12, pad=15)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {summary_path}")

    # --- 2. BAR PLOT (Feature Importance) ---
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=max_features, show=False)
    plt.title("SHAP Feature Importance", fontsize=12, pad=15)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bar_path}")

def save_model_report(y_true, y_pred, y_prob, model, X_test, output_path="MODEL_REPORT.md"):
    """Generates global performance metrics, classification reports, confusion matrix, and SHAP plots."""
    
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc_score = roc_auc_score(y_true, y_prob)

    markdown_content = f"""# 📊 XGBoost Model Evaluation Report

## 1. Global Performance Metrics
- **ROC AUC:** `{auc_score:.4f}`
- **Accuracy:** `{report_dict['accuracy']:.4f}`
- **Macro F1-Score:** `{report_dict['macro avg']['f1-score']:.4f}`

---

## 2. Classification Metrics by Class
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (No Readmission)** | {report_dict['0']['precision']:.4f} | {report_dict['0']['recall']:.4f} | {report_dict['0']['f1-score']:.4f} | {report_dict['0']['support']} |
| **1 (Readmitted <30 Days)** | {report_dict['1']['precision']:.4f} | {report_dict['1']['recall']:.4f} | {report_dict['1']['f1-score']:.4f} | {report_dict['1']['support']} |

---

## 3. Confusion Matrix Breakdown
- **True Negatives (TN):** `{tn}`
- **False Positives (FP):** `{fp}`
- **False Negatives (FN):** `{fn}`
- **True Positives (TP):** `{tp}`
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Generating SHAP values and plots
    print("Generating SHAP values and plots...")
    shap_values = generate_shap_values(model, X_test)
    save_shap_plots(shap_values, X_test, output_dir="reports")

    # Adding SHAP plots section to the report
    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\n---\n")
        f.write("## 4. Model Interpretability (SHAP Analysis)\n")
        f.write("Plots below show the global and directional impact of features on the model's decisions:\n\n")
        f.write("### Summary Plot (Directional Impact)\n")
        f.write("![SHAP Summary Plot](reports/shap_summary.png)\n\n")
        f.write("### Feature Importance (Global Importance)\n")
        f.write("![SHAP Bar Plot](reports/shap_bar.png)\n")

    print(f"✅ Report saved successfully to `{output_path}`")