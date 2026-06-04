import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Telco Churn Core Inference Engine",
    description="Production endpoint serving custom 3-layer ANN predictions to Streamlit frontend",
    version="4.0"
)

MODEL = None
SCALER = None

# 2. Strict blueprint of the dummy columns your neural network expects.
# (This reordering layout ensures your matrix shape matches your notebook training phase exactly)
EXPECTED_TRAINING_COLUMNS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaperlessBilling_Yes", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
]

# 3. Load ML Artifacts into memory on Server Startup
@app.on_event("startup")
def load_artifacts():
    global MODEL, SCALER
    try:
        SCALER = joblib.load("scaler.pkl")
        MODEL = tf.keras.models.load_model("final_churn_ann.keras")
        print("🎉 Neural Network and Scaler loaded into server memory successfully!")
    except Exception as e:
        print(f"❌ Critical Error loading weights or parameters: {str(e)}")


# 4. Data Transfer Object matching Streamlit widget string properties exactly
class StreamlitCustomerPayload(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# 5. Internal Matrix Processing Pipeline
def transform_streamlit_payload(raw_df: pd.DataFrame) -> np.ndarray:
    """
    Transforms the single customer row dictionary into dummy features,
    normalizes numerical parameters, and aligns matrices perfectly for TensorFlow.
    """
    # The categorical text targets collected by the streamlit sidebar UI
    cat_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    # Run identical drop_first dummy encoding parameters
    df_encoded = pd.get_dummies(raw_df, columns=cat_cols, drop_first=True)

    # Scale numerical entries via our saved scaler parameters
    num_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    df_encoded[num_features] = SCALER.transform(df_encoded[num_features])

    # Build columns if missing (e.g. if a user selects month-to-month, one-year column is appended as 0)
    for col in EXPECTED_TRAINING_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Rearrange matrix indexing to match neural layout template
    df_final = df_encoded[EXPECTED_TRAINING_COLUMNS]

    return df_final.values.astype(np.float32)


# 6. Production Prediction Route
@app.post("/predict")
def predict_churn(payload: StreamlitCustomerPayload):
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Server components initializing.")

    try:
        # Convert Pydantic item to Pandas Dataframe
        raw_df = pd.DataFrame([payload.dict()])

        # Parse text values and structure data
        processed_features = transform_streamlit_payload(raw_df)

        # Pass vector straight to the TensorFlow hidden dense layers
        raw_probability = float(MODEL.predict(processed_features)[0][0])

        # Apply Andrew Ng's Optimized High-Recall Threshold
        custom_threshold = 0.35
        prediction_label = 1 if raw_probability >= custom_threshold else 0

        return {
            "status": "success",
            "ann_churn_probability": round(raw_probability, 4),
            "ann_prediction_label": prediction_label,
            "ann_prediction_text": "Yes" if prediction_label == 1 else "No",
            "applied_threshold": custom_threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Core Processing Exception: {str(e)}")

# 7. Basic Server Route
@app.get("/")
def roots():
    return {"status": "active", "engine": "FastAPI-TensorFlow-Inference"}
