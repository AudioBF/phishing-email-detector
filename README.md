# Phishing Email Detector

A phishing email detector using BERT-tiny for text analysis, optimized for low memory consumption and high performance.

## 🚀 Features

- Modern and responsive web interface
- Real-time email analysis
- Email examples for testing
- High-precision phishing detection
- Confidence and probability display
- Light/dark mode
- Word count
- Optimized for low memory consumption

## 🎯 How to Use

### Web Interface
1. Access the application at `http://localhost:8001` (local) or your deployment URL
2. You will see an interface with:
   - Text field for email input
   - Pre-loaded examples selector
   - Analysis button
   - Word counter
   - Light/dark mode toggle

### Email Analysis
1. **Method 1 - Direct Text**:
   - Paste the email text in the main field
   - Click "Check Email"
   - Wait for analysis (usually 1-2 seconds)
   - View results with confidence and probability

2. **Method 2 - Examples**:
   - Select an example from the dropdown
   - Text will be automatically filled
   - Click "Check Email"
   - Analyze the result

### Results
The analysis returns:
- Classification (Spam/Ham)
- Confidence level
- Probability
- Text characteristics

### API Usage Example
```python
import requests

# API URL
url = "https://phishing-email-detector-9548433b70c2.herokuapp.com/email/analyze"

# Email for analysis
email_text = """
Subject: Urgent: Your Account Security Alert

Dear Customer,

We have detected unusual activity on your account. Please verify your identity immediately by clicking the link below:

http://suspicious-link.com/verify

Failure to respond within 24 hours will result in account suspension.

Best regards,
Security Team
"""

# Send request
response = requests.post(url, json={"email_text": email_text})
result = response.json()

print(f"Is phishing? {result['is_spam']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probability: {result['probability']:.2%}")
```

### Email Examples
The system includes examples of:
- Phishing emails
- Legitimate emails
- Emails with suspicious characteristics
- Emails with false urgency

### Usage Tips
1. **For Better Accuracy**:
   - Include email subject
   - Keep original text
   - Avoid very short texts

2. **Interpreting Results**:
   - Confidence > 80%: High probability of accuracy
   - Confidence 50-80%: Requires additional analysis
   - Confidence < 50%: Likely legitimate

3. **Performance Optimization**:
   - Use prediction cache
   - Limit text size
   - Use pre-loaded examples

## 🏗️ Project Structure

```
phishing-email-detector/
├── app/
│   ├── main.py          # FastAPI API and routes
│   ├── predict.py       # Optimized prediction model
│   ├── bert_classifier.py # BERT classifier implementation
│   └── routers/         # API routes
├── static/
│   ├── index.html       # Web interface
│   ├── style.css        # Styles
│   ├── script.js        # Frontend logic
│   ├── emails_samples.json # Email examples
│   └── assets/          # Static resources
├── models/
│   └── best_model.pt    # Trained model
├── nltk_data/           # NLTK data
├── requirements.txt     # Dependencies
└── Procfile            # Deployment configuration
```

## 📋 Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt`:
  - FastAPI 0.104.1
  - Uvicorn 0.24.0
  - Transformers 4.35.2
  - PyTorch 2.1.2+cpu
  - NLTK 3.8.1
  - Other dependencies listed in the file

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Audiobf/phishing-email-detector.git
cd phishing-email-detector
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Local Execution

1. Start the server:
```bash
uvicorn app.main:app --reload --port 8001
```

2. Access the application at:
```
http://localhost:8001
```

## 🔍 Technical Details

### BERT Model
- Uses BERT-tiny for efficiency
- CPU-optimized implementation
- Reduced fully connected layers
- Dropout for regularization

### Frontend
- Responsive interface
- Light/dark theme
- Real-time word count
- Visual analysis feedback
- Pre-loaded examples

### Optimizations
- Prediction cache
- Automatic memory cleanup
- On-demand loading
- Text size limit
- Singleton pattern

## 🤝 Contributing

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT

## ⚠️ Limitations

- Maximum slug size: 300MB (soft limit)
- Available memory: 512MB (basic dyno)
- Maximum text size: 1000 characters
- Prediction cache: last 100 predictions