from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal


class Customer(BaseModel):
    # numeric
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int

    # binary (0/1 ONLY)
    Partner: int = Field(..., ge=0, le=1)
    Dependents: int = Field(..., ge=0, le=1)
    PhoneService: int = Field(..., ge=0, le=1)
    PaperlessBilling: int = Field(..., ge=0, le=1)

    # categorical (OHE)
    gender: Literal["Male", "Female"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]

    OnlineSecurity: Literal["Yes","No","No internet service"]
    OnlineBackup: Literal["Yes","No","No internet service"]
    DeviceProtection: Literal["Yes","No","No internet service"]
    TechSupport: Literal["Yes","No","No internet service"]
    StreamingTV: Literal["Yes","No","No internet service"]
    StreamingMovies: Literal["Yes","No","No internet service"]

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

app = FastAPI(title="Customer Churn Prediction API")
model = joblib.load("churn_model.pkl")

@app.get("/")
def home():
    return{"status":"running"}

@app.post("/predict")
def predict(data:Customer):
    input_df = pd.DataFrame([data.dict()])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    return{"churn_prediction":int(pred),
           "chun_probability": round(float(prob),4)
           }
