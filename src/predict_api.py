import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import xgboost as xgb

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Readmission Prediction Service",
    description="API to predict the likelihood of a patient being readmitted to the hospital within 30 days after discharge, based on a trained XGBoost model registered in MLflow.",
    version="1.0.0"
)

# Pydantic model for input data validation
class PatientData(BaseModel):
    time_in_hospital: int = Field(..., example=3, description="Time spent in hospital (days)")
    num_lab_procedures: int = Field(..., example=41, description="Number of laboratory tests performed")
    num_procedures: int = Field(..., example=0, description="Number of medical procedures (excluding lab)")
    num_medications: int = Field(..., example=15, description="Number of medications prescribed")
    
    features_vector: list[float] = Field(
        ..., 
        example=[
            3.0, 41.0, 0.0, 15.0,         # 4 numeric features
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, # race OHE
            1.0, 0.0,                     # gender OHE
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # age OHE
            1.0, 0.0,                     # change OHE
            1.0, 0.0                      # diabetesMed OHE
        ], 
        description="Vector of numeric and one-hot encoded categorical features (Exactly 26 features)"
    )

# Define paths for model loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model")
MODEL_FILE = os.path.join(MODEL_PATH, "model.ubj")

model = None
feature_names = None

# Model loading logic with enhanced error handling and feature name extraction
try:
    print(f"⏳ Loading native XGBoost model from path: {MODEL_FILE}...")
    if os.path.exists(MODEL_FILE):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        print("✅ Model loaded successfully using native XGBoost loader!")
        
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
            print(f"📋 Successfully extracted {len(feature_names)} feature names.")
    else:
        print(f"❌ Error: Model file not found at {MODEL_FILE}")
except Exception as e:
    print(f"❌ Error while loading the model natively: {e}")
    model = None


# Endpoints

@app.get("/", summary="Health check endpoint")
def read_root():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", summary="Predict readmission")
def predict_readmission(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded properly on the server.")
    
    if feature_names and len(patient.features_vector) != len(feature_names):
        raise HTTPException(
            status_code=400, 
            detail=f"Feature shape mismatch. Model expects {len(feature_names)} features, but got {len(patient.features_vector)}."
        )
    
    try:
        input_matrix = np.array([patient.features_vector])
        
        if feature_names:
            input_data = pd.DataFrame(input_matrix, columns=feature_names)
        else:
            input_data = input_matrix
        
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        readmission_probability = float(probabilities[0][1])
        will_be_readmitted = int(prediction[0])
        
        return {
            "readmitted_prediction": will_be_readmitted,
            "readmission_probability": round(readmission_probability, 4),
            "message": "Patient likely to be readmitted within 30 days." if will_be_readmitted == 1 else "Low risk of rapid readmission."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")