from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.predict import predict_email

router = APIRouter()

class EmailRequest(BaseModel):
    text: str

@router.post("/analyze")
async def analyze_email(request: EmailRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Email text cannot be empty")
        
        result = predict_email(request.text)
        return {
            "is_spam": result["is_spam"],
            "confidence": round(result["confidence"] * 100, 2),
            "probability": round(result["probability"] * 100, 2),
            "features": result["features"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 