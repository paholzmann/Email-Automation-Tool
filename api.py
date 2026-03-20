from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
baseline_model = joblib.load("models/baseline_model.pkl")

class Request(BaseModel):
    text: str

@app.post("/predict")
def predict(request: Request):
    pred = baseline_model.predict([request.text])[0]
    return {"prediction": pred}