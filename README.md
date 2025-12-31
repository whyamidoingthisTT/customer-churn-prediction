# 📊 Customer Churn Prediction System

## 🔍 Overview

This project predicts whether a telecom customer is likely to **churn (leave the service)** based on customer demographics, service subscriptions, and contract details.

It is an **end-to-end machine learning project** that covers:
- Data preprocessing
- Feature engineering
- Model training & evaluation
- Model persistence
- Real-time inference via a FastAPI service

Special emphasis is placed on **correct schema handling**, **ternary categorical features**, and **production-safe inference**.

---

## 🧠 Problem Statement

Customer churn is a critical issue for subscription-based businesses.  
The goal of this project is to:

> **Predict customer churn using historical customer data to identify high-risk users early.**

This enables businesses to:
- Reduce customer attrition
- Design targeted retention strategies
- Improve customer lifetime value

---

## 📁 Dataset

- **File:** `churn_dataset.csv`
- **Type:** Structured tabular dataset
- **Target Variable:** `Churn`

### Feature Categories

| Feature Type | Examples |
|-------------|---------|
| Numeric | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Binary (Yes / No) | `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` |
| Ternary Categorical | `OnlineSecurity`, `TechSupport`, `StreamingTV` (`Yes / No / No internet service`) |
| Multi-class Categorical | `Contract`, `InternetService`, `PaymentMethod` |

⚠️ A major challenge addressed in this project was **correctly handling ternary categorical features**, which are often mistakenly treated as binary.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Removed non-informative identifier columns
- Explicitly encoded **only true binary features**
- Preserved ternary categorical features for proper encoding
- Handled missing values (`TotalCharges`)
- Ensured schema consistency between training and inference

### 2️⃣ Feature Engineering
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- Binary features passed through without distortion

### 3️⃣ Model Training
- **Algorithm:** Random Forest Classifier
- **Trees:** 200
- Trained using a `Pipeline` with `ColumnTransformer` to ensure reproducibility

### 4️⃣ Evaluation
- Accuracy and classification report
- Feature importance visualization to understand churn drivers

---

## 🤖 Model Output

The model returns:

- **`churn_prediction`**
  - `0` → Customer unlikely to churn
  - `1` → Customer likely to churn
- **`churn_probability`**
  - Value between `0` and `1`
  - Indicates likelihood of churn

Predictions are probabilistic and based on learned patterns, not memorization of dataset rows.

---

## 🚀 FastAPI Inference Service

The trained model is deployed using **FastAPI** for real-time predictions.

### Endpoint

POST `/predict`

### Example Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": 0,
  "Dependents": 0,
  "tenure": 2,
  "PhoneService": 1,
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": 1,
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 99.85,
  "TotalCharges": 199.7
}
```

### Example Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.72
}
```

---

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Random Forest Classifier
- FastAPI
- Uvicorn
- Joblib
- Jupyter Notebook
- Git & GitHub

---

## ⚠️ Key Challenges Addressed

- Incorrect assumptions about binary vs ternary features
- Schema mismatch between training and inference pipelines
- Consistent preprocessing across notebook and API
- Safe handling of categorical values in production
- Git history recovery using reflog and rebase resolution

---

## ▶️ How to Run the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API server

```bash
uvicorn main:app --reload
```

### Open Swagger UI

http://127.0.0.1:8000/docs

---

## 📌 Future Enhancements

- Handle class imbalance explicitly
- Add SHAP-based explainability
- Implement batch prediction endpoints
- Containerize using Docker
- Deploy on cloud platforms

---

## ⭐ Final Note

This project emphasizes **data correctness, schema alignment, and production readiness**, reflecting real-world ML engineering challenges.
