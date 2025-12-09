from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

class Customer(BaseModel):
    #includes every feature used in the model
    tenure: int #
    MonthlyCharges: float
    TotalCharges: float
    Churn: int
    gender: str #
    SeniorCitizen: int #
    Partner: int #
    Dependents: int #
    PhoneService: int
    MultipleLines: str
    InternetServices: str
    OnlineSecurity: int
    Contract: str
    PaperlessBilling: int
    PaymentMethod: str


app = FastAPI()
model = joblib.load("churn_model.pkl")

@app.get("/")
def home():
    return{"status":"running"}

@app.post("/predict")
def predict(data:dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return{"churn_prediction":int(pred)}
