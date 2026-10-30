import os
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature  
import optuna
from pyspark.sql import SparkSession
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd

from generate_report import save_model_report  # Importing the function to save model evaluation report

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

# Loading data for Optuna hyperparameter tuning (X is a Pandas DataFrame)
X, y = load_processed_data_with_feature_names()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_weight = neg_count / pos_count
print(f"Calculated scale_pos_weight for imbalanced classes: {scale_weight:.2f}")

def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    
    # Defining the hyperparameter search space
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "scale_pos_weight": scale_weight,  # Using the calculated scale_pos_weight for imbalanced classes   
        "use_label_encoder": False,
        "eval_metric": "aucpr" # Using AUC-PR as the evaluation metric for imbalanced classification
    }
    
    # Logging the trial in MLflow
    with mlflow.start_run(run_name=f"Optuna_Trial_{trial.number}", nested=True):
        # Automatically log parameters and metrics for this trial
        mlflow.log_params(params)
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        pred_proba = model.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(y_test, pred_proba)
        mlflow.log_metric("pr_auc", pr_auc)
        
        return pr_auc

def find_optimal_threshold(y_true, y_prob):
    """Find the optimal threshold for converting predicted probabilities to binary class labels based on F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0
    
    # Testing thresholds from 0.3 to 0.85 in increments of 0.02 to find the one that maximizes F1 score
    for th in np.arange(0.3, 0.85, 0.02):
        preds = (y_prob >= th).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = th
            
    return best_threshold

def run_hyperparameter_tuning():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Diabetes_Readmission_With_Optuna")
    
    # Main run for hyperparameter tuning
    with mlflow.start_run(run_name="Optuna_Hyperparameter_Tuning") as parent_run:
        
        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10) # for time-saving, we limit to 10 trials
        
        print(f"🏆 Best parameters: {study.best_params}")
        print(f"📈 Best PR AUC: {study.best_value:.4f}")
        
        # Save the best parameters and metrics to MLflow
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)
        
        # Training the best model on the entire training set and logging it to MLflow
        best_model = XGBClassifier(**study.best_params, 
                                   scale_pos_weight=scale_weight,  # Ensure the best model uses the correct scale_pos_weight
                                   use_label_encoder=False, 
                                   eval_metric="logloss")
        best_model.fit(X_train, y_train)
        
        
        pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Finding the optimal threshold for converting predicted probabilities to binary class labels
        optimal_threshold = find_optimal_threshold(y_test, pred_proba)
        print(f"Optimal threshold for classification based on F1 score: {optimal_threshold:.2f}")

        predictions = (pred_proba >= optimal_threshold).astype(int)

        signature = infer_signature(X_train, predictions)
        
        # Signature ensures that the input schema is preserved in MLflow UI, which is crucial for production deployment and API validation.
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="best_diabetes_model",
            registered_model_name="DiabetesReadmissionModel",
            signature=signature # <-- Schema is preserved in MLflow UI
        )
        print("✅ Best model with signature and feature names has been registered in MLflow Registry!")

        save_model_report(
            y_true=y_test,
            y_pred=predictions,
            y_prob=pred_proba,
            output_path="MODEL_REPORT.md"
        )

if __name__ == "__main__":
    run_hyperparameter_tuning()