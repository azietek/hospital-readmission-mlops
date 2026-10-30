# End-to-End Hospital Readmission Prediction MLOps Pipeline

An enterprise-grade Machine Learning Engineering project demonstrating the operationalization of a clinical prediction pipeline. This project transitions a standard data science problem—predicting diabetic patient readmission using the **Diabetes 130-US Hospitals Dataset**—into a production-ready, containerized microservice equipped with automated hyperparameter tuning, experiment tracking, and CI/CD.

## 🎯 Project Objective
To bridge the gap between notebook-based experimentation and robust, maintainable production engineering by implementing clean code standards, automated pipelines, and continuous integration.

## 🏗️ Architecture & Tech Stack



- **Data Engineering (ETL):** **PySpark & Spark SQL** used for distributed data ingestion, cleaning, and scalable feature transformations (`StringIndexer`, `OneHotEncoder`, `VectorAssembler`).
- **Experimentation & Hyperparameter Tuning:** **Optuna** paired with **MLflow** for automated tracking of trials, metrics (ROC AUC, Accuracy), and automated artifacts/model logging.
- **Model Registry & Versioning:** **MLflow Model Registry** for centralizing model versions.
- **ML Modeling:** **XGBoost Classifier** optimized for clinical classification metrics.
- **Deployment & Serving:** **FastAPI & Uvicorn** delivering a high-performance async ML inference microservice with request validation via **Pydantic**.
- **Containerization:** **Docker** ensuring reproducible runtime environments across staging and production.
- **CI/CD & Code Quality:** **GitHub Actions** and **PyTest** for automated execution of unit tests upon code change.

## 📁 Repository Structure
```text
hospital-readmission-mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions CI Automation
├── data/                      # Source & intermediate datasets (.parquet format)
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py      # PySpark ETL and Feature Engineering pipeline
│   ├── train.py               # Optuna tuning & MLflow experiment/artifact logging
│   ├── predict_api.py         # Production FastAPI model serving script
│   └── best_model/            # Downloaded production model deployment artifact
├── tests/
│   └── test_pipeline.py       # Unit tests for API structure and data logic
├── Dockerfile                 # Production multi-stage container build definition
├── requirements.txt           # Explicitly pinned project dependencies
└── README.md                  # Comprehensive technical documentation
```

## 🚀 Getting Started

### 1. Environment Setup (macOS / Linux)

Ensure you have Python 3.10+ installed. Clone this repository and initialize an isolated virtual environment:

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

Every `push` or `pull_request` to the main branch automatically triggers the `.github/workflows/ci-cd.yml` workflow pipeline on GitHub runners, validating environment setup, dependency resolution, and test execution.