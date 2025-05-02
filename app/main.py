from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import logging
import os
import sys
import json
import nltk
from app.routers import email_analyzer

# Adiciona o diretório atual ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predict import predict_email

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Phishing Email Detector API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the static directory
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(email_analyzer.router)

# Define request models
class EmailRequest(BaseModel):
    text: str

@app.get("/")
async def read_root():
    """Root route that serves the index.html file"""
    try:
        return FileResponse(os.path.join(static_dir, "index.html"))
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading interface")

@app.get("/samples")
async def get_samples():
    """Route to get email samples"""
    try:
        samples_path = os.path.join(static_dir, "emails_samples.json")
        with open(samples_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        logger.info(f"Returning {len(samples['emails'])} email samples")
        return samples
    except Exception as e:
        logger.error(f"Error loading samples: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading email samples")

@app.post("/analyze")
async def analyze_email(request: EmailRequest):
    """Route to analyze email content"""
    try:
        if not request.text.strip():
            logger.warning("Empty text received")
            raise HTTPException(status_code=400, detail="Email text cannot be empty")
        
        result = predict_email(request.text)
        logger.info(f"Analysis completed: {'Spam' if result['is_spam'] else 'Ham'} (confidence: {result['confidence']})")
        
        return {
            "is_spam": result["is_spam"],
            "confidence": round(result["confidence"] * 100, 2),
            "probability": round(result["probability"] * 100, 2),
            "features": result["features"]
        }
    except Exception as e:
        logger.error(f"Error analyzing email: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Download required NLTK data
try:
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)
    
    # Download punkt if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir)
    
    # Download stopwords if not already downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_dir)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
