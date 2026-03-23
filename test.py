from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()
@app.get("/health")
def health_check():
    """
    curl http://127.0.0.1:8000/health
    """
    return {"status": "ok"}

@app.get("/emails/{email_id}")
def get_email(email_id: int):
    """
    curl http://127.0.0.1:8000/emails/1
    """
    if email_id != 1:
        raise HTTPException(status_code=404, detail=f"Email with id {email_id} not found")
    return {"email_id": email_id}

class EmailRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1)

class PredictionResponse(BaseModel):
    category: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict_mail(data: EmailRequest):
    """
    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"subject\": \"Refund request\", \"body\": \"I want a refund\"}"
    """
    return {
        "category": "Billing",
        "confidence": 0.91
    }