# 🏥 Hospital Readmission MLOps Pipeline

Production-ready MLOps pipeline for predicting 30-day hospital readmissions using the **Diabetes 130-US Hospitals (1999–2008)** dataset. The project focuses on robust PySpark ETL, domain-specific feature engineering, clinical interpretability via SHAP, and rigorous tracking.

---

## 🚀 Key Results & Performance
Our latest XGBoost model successfully balances clinical recall and precision, achieving strong generalization on high-cardiac/diabetic risk profiles:

* **ROC-AUC:** `0.6661`
* **Recall (Class 1 - Readmitted <30 Days):** `57.16%` (prioritizing high-risk patient capture)
* **Accuracy:** `66.25%`

---

## 🛠️ Tech Stack & Architecture
- **Data Processing & ETL:** PySpark (handling 100k+ hospital records)
- **Machine Learning:** XGBoost Classifier
- **Experiment Tracking & Registry:** MLflow
- **Model Interpretability:** SHAP (`summary_plot` & `bar_plot`)
- **Pipeline Orchestration:** Python modular architecture (`src/data_ingestion.py`, `src/train.py`, etc.)

---

## 🧬 Domain-Specific Feature Engineering & ETL Decisions
Unlike standard black-box ML pipelines, this project embeds strict clinical logic:
1. **Mortality & Hospice Filtering:** Removed records with discharge disposition codes corresponding to death (`11`) and hospice care (`13, 14, 19, 20`) to prevent data leakage and target misalignment.
2. **Discharge Disposition Grouping:** Consolidated dozens of fragmented hospital discharge codes into meaningful clinical categories (`Home`, `Transfer_Rehab`, `Home_Health`) to improve feature stability and SHAP legibility.
3. **Continuous Age Transformation:** Converted binned age ranges into continuous midpoint values (`age_numeric`) for smoother tree splits.
4. **Utilization Aggregates:** Engineered high-level activity metrics (e.g., total prior visits, intensity rates) to reduce redundancy in raw lab/procedure counts.

---

## 📂 Repository Structure
```text
hospital-readmission-mlops/
├── data/                  # Raw and processed datasets
├── reports/               # Generated SHAP plots (summary & bar) & evaluation reports
├── src/                   # Core pipeline scripts (ETL, training, evaluation)
├── mlruns/                # MLflow tracking directory
├── MODEL_REPORT.md        # Detailed latest evaluation markdown report
└── README.md              # Project documentation
```

## 🚀 Getting Started

### 1. Environment Setup (macOS / Linux)

Ensure you have Python 3.12+ installed. Clone this repository and initialize an isolated virtual environment:

```
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install exact dependency versions
pip install -r requirements.txt
```

### 2. Running Data Engineering & Feature Pipeline

Execute the PySpark pipeline to clean the raw hospital dataset, apply categorical encoding, and serialize the output into high-performance Apache Parquet format:

```
python src/data_ingestion.py
```

### 3. Hyperparameter Optimization & Tracking (MLflow + Optuna)

Start the local MLflow tracking server in your terminal:

```
mlflow server --host 127.0.0.1 --port 5000
```

In a separate terminal window, trigger the automated training loop. Optuna will search the parameter space, log each trial as a nested run inside MLflow, and automatically register the champion model to the registry:

```
python src/train.py
```

Open `http://127.0.0.1:5000` in your browser to view the interactive MLOps dashboard.

### 4. Running the Local API Microservice

Before serving, download the latest production-ready model artifact directly from the MLflow backend:

```
python -c "import mlflow; mlflow.set_tracking_uri('[http://127.0.0.1:5000](http://127.0.0.1:5000)'); mlflow.artifacts.download_artifacts(artifact_uri='models:/DiabetesReadmissionModel/latest', dst_path='src/best_model')"
```

Fire up the FastAPI production server locally:

```
uvicorn src.predict_api:app --reload
```

Navigate to `http://127.0.0.1:8000/docs` to interact with the auto-generated Swagger UI and test the `/predict` endpoint with sample patient data payloads.

### 5. Production Containerization (Docker)

Build the containerized microservice package which embeds the local PySpark-processed model binary:

```
docker build -t diabetes-api .
```

Run the isolated container mapping port 8000:

```
docker run -p 8000:8000 diabetes-api
```

## 🧪 Quality Assurance & Continuous Integration (CI/CD)

The project utilizes `pytest` to guarantee endpoint schema integrity. Run the test suite locally via:

```
pytest
```

---
## 📄 Full Project Documentation
For a deep dive into the methodology, feature engineering decisions, and clinical interpretation of SHAP values, check out the [Full Project Report](PROJECT_REPORT.md).

