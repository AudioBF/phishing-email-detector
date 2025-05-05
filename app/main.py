from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging
import os
import sys
import json
import nltk
from app.routers import email_analyzer

# Adiciona o diretório atual ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure NLTK data path
nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
nltk.data.path.append(nltk_data_dir)

# Verify NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    logger.info("NLTK resources loaded successfully")
except LookupError as e:
    logger.error(f"NLTK resource not found: {e}")
    raise

from app.predict import predict_email

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
app.include_router(email_analyzer.router, prefix="/email", tags=["email"])

# Define request models
class EmailRequest(BaseModel):
    email_text: str

class EmailResponse(BaseModel):
    is_spam: bool
    confidence: float
    probability: float
    features: dict

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
        logger.info(f"Loading samples from: {samples_path}")
        
        if not os.path.exists(samples_path):
            logger.error(f"File not found: {samples_path}")
            raise HTTPException(status_code=404, detail="Email samples file not found")
        
        try:
            with open(samples_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {str(e)}")
            raise HTTPException(status_code=500, detail="Invalid JSON format in samples file")
        
        if not isinstance(samples, dict) or 'emails' not in samples:
            logger.error("Invalid samples file format")
            raise HTTPException(status_code=500, detail="Invalid samples file format")
        
        # Verify the structure of each email
        for i, email in enumerate(samples['emails']):
            if not all(key in email for key in ['title', 'type', 'email']):
                logger.error(f"Invalid email format at index {i}")
                raise HTTPException(status_code=500, detail=f"Invalid email format at index {i}")
        
        return samples
    except Exception as e:
        logger.error(f"Unexpected error in get_samples: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/static/emails_samples.json")
async def get_samples_file():
    """Serve the email samples JSON file directly"""
    try:
        samples_path = os.path.join(static_dir, "emails_samples.json")
        logger.info(f"Serving samples file from: {samples_path}")
        
        if not os.path.exists(samples_path):
            logger.error(f"File not found: {samples_path}")
            raise HTTPException(status_code=404, detail="Email samples file not found")
        
        return FileResponse(samples_path, media_type="application/json")
    except Exception as e:
        logger.error(f"Error serving samples file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=EmailResponse)
async def analyze_email(request: EmailRequest):
    """Route to analyze email content"""
    try:
        if not request.email_text.strip():
            logger.warning("Empty text received")
            raise HTTPException(status_code=400, detail="Email text cannot be empty")
        
        result = predict_email(request.email_text)
        logger.info(f"Analysis completed: {'Spam' if result['is_spam'] else 'Ham'} (confidence: {result['confidence']})")
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing email: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(static_dir, "assets", "favicon.ico"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
