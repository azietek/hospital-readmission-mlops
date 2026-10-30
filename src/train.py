import os
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature  # <-- Dodane importowanie sygnatury
import optuna
from pyspark.sql import SparkSession
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd

def load_processed_data_with_feature_names():
    spark = SparkSession.builder.appName("DiabetesTraining").master("local[*]").getOrCreate()
    df = spark.read.parquet("data/processed.parquet")

    # Extracting feature names from Spark DataFrame metadata to ensure correct order for XGBoost input
    schema_fields = df.schema["features"].metadata["ml_attr"]["attrs"]
    
    feature_names = []
    all_attrs = []
    if "numeric" in schema_fields:
        all_attrs.extend(schema_fields["numeric"])
    if "binary" in schema_fields:
        all_attrs.extend(schema_fields["binary"])
    if "nominal" in schema_fields:
        all_attrs.extend(schema_fields["nominal"])
        
    all_attrs = sorted(all_attrs, key=lambda x: x["idx"])
    feature_names = [attr["name"] for attr in all_attrs]
    
    rows = df.collect()
    spark.stop()
    
    X_matrix = np.array([row["features"].toArray() for row in rows])
    y_vector = np.array([row["label"] for row in rows])
    
    # Cleaning feature names to ensure compatibility with Pandas DataFrame column naming conventions
    # Exchanging '[', ']', and '<' with '_' to avoid issues in Pandas DataFrame column names
    cleaned_feature_names = []
    for name in feature_names:
        cleaned_name = name.replace("[", "_").replace("]", "_").replace("<", "less_than_")
        cleaned_feature_names.append(cleaned_name)
    
    # Creating a Pandas DataFrame with cleaned feature names for better interpretability and compatibility
    X_df = pd.DataFrame(X_matrix, columns=cleaned_feature_names)
    
    return X_df, y_vector

# Loading data for Optuna hyperparameter tuning (X jest teraz obiektem Pandas DataFrame)
X, y = load_processed_data_with_feature_names()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    
    # Defining the hyperparameter search space
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "use_label_encoder": False,
        "eval_metric": "aucpr" # Using AUC-PR as the evaluation metric for imbalanced classification
    }
    
    # Logging the trial in MLflow
    with mlflow.start_run(run_name=f"Optuna_Trial_{trial.number}", nested=True):
        # Automatically log parameters and metrics for this trial
        mlflow.log_params(params)
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, pred_proba)
        
        # Logging the ROC AUC metric for this trial
        mlflow.log_metric("roc_auc", auc)
        
        return auc

def run_hyperparameter_tuning():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Diabetes_Readmission_With_Optuna")
    
    # Main run for hyperparameter tuning
    with mlflow.start_run(run_name="Optuna_Hyperparameter_Tuning") as parent_run:
        
        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5) # for time-saving, we limit to 5 trials
        
        print(f"🏆 Best parameters: {study.best_params}")
        print(f"📈 Best ROC AUC: {study.best_value:.4f}")
        
        # Save the best parameters and metrics to MLflow
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)
        
        # Training the best model on the entire training set and logging it to MLflow
        best_model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric="logloss")
        best_model.fit(X_train, y_train)
        
        # Automatically log the model with MLflow, including the feature names and signature
        predictions = best_model.predict(X_test)
        signature = infer_signature(X_train, predictions)
        
        # Signature ensures that the input schema is preserved in MLflow UI, which is crucial for production deployment and API validation.
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="best_diabetes_model",
            registered_model_name="DiabetesReadmissionModel",
            signature=signature # <-- Schema is preserved in MLflow UI
        )
        print("✅ Best model with signature and feature names has been registered in MLflow Registry!")

if __name__ == "__main__":
    run_hyperparameter_tuning()