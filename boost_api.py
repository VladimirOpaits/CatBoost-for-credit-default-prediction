from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostClassifier
import uvicorn
import pandas as pd


model = CatBoostClassifier()
model.load_model("catboost_credit_model.cbm")

class ClientData(BaseModel):
    LIMIT_BAL: int
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: int
    BILL_AMT2: int
    BILL_AMT3: int
    BILL_AMT4: int
    BILL_AMT5: int
    BILL_AMT6: int
    PAY_AMT1: int
    PAY_AMT2: int
    PAY_AMT3: int
    PAY_AMT4: int
    PAY_AMT5: int
    PAY_AMT6: int


app = FastAPI(title="Credit Default Predictor")

@app.post("/predict")
def predict(client: ClientData):
    X = pd.DataFrame([client.dict()])
    
    cat_features = [
        "SEX", "EDUCATION", "MARRIAGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
    ]
    
    proba = model.predict_proba(X)[:, 1][0]
    pred = int(proba > 0.5)
    
    return {"prediction": pred, "probability": proba}