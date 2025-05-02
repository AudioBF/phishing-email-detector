import torch
import json
import numpy as np
from app.preprocess import TextPreprocessor
from app.train import EmailClassifier
import logging
import os
from pathlib import Path
from app.predict import EmailPredictor
from app.validate import cross_validate
from transformers import BertTokenizer
from app.bert_classifier import SpamClassifier
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(file_path='data/test_emails.json'):
    """Load test data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['emails']

def test_model(model, scaler, test_data):
    """Test the model on the test dataset"""
    preprocessor = TextPreprocessor()
    correct = 0
    total = 0
    
    logger.info("\nTesting model on new dataset...")
    logger.info("-" * 50)
    
    for email in test_data:
        # Process email
        processed_text, features = preprocessor.process_email(email['text'])
        features_array = np.array(list(features.values())).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Get prediction
        with torch.no_grad():
            output = model(features_tensor)
            prediction = (output > 0.5).float().item()
        
        # Get true label
        true_label = 1 if email['label'] == 'spam' else 0
        
        # Update counts
        total += 1
        if prediction == true_label:
            correct += 1
        
        # Log results
        result = "✓" if prediction == true_label else "✗"
        logger.info(f"{result} Email: {email['text'][:100]}...")
        logger.info(f"   True Label: {'spam' if true_label else 'ham'}")
        logger.info(f"   Prediction: {'spam' if prediction else 'ham'}")
        logger.info(f"   Confidence: {output.item():.2f}")
        logger.info("-" * 50)
    
    # Calculate and log accuracy
    accuracy = 100 * correct / total
    logger.info(f"\nFinal Results:")
    logger.info(f"Total Emails: {total}")
    logger.info(f"Correct Predictions: {correct}")
    logger.info(f"Accuracy: {accuracy:.2f}%")

def test_predictor():
    """Testa o predictor com exemplos de spam e ham"""
    logger.info("Iniciando testes do predictor...")
    
    # Exemplos de teste
    test_emails = {
        "spam": [
            "Ganhe R$ 1.000.000 agora! Clique aqui para receber seu prêmio!",
            "URGENTE: Sua conta será bloqueada se não atualizar seus dados agora!",
            "Parabéns! Você ganhou um iPhone grátis! Responda para receber!"
        ],
        "ham": [
            "Olá, gostaria de agendar uma reunião para amanhã às 14h.",
            "Segue em anexo o relatório mensal solicitado.",
            "Bom dia, tudo bem? Preciso de ajuda com o projeto."
        ]
    }
    
    # Inicializar predictor
    try:
        predictor = EmailPredictor()
        logger.info("Predictor inicializado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao inicializar predictor: {str(e)}")
        return
    
    # Testar predições
    results = {"spam": [], "ham": []}
    
    for label, emails in test_emails.items():
        for email in emails:
            try:
                result = predictor.predict(email)
                results[label].append({
                    "email": email,
                    "is_spam": result["is_spam"],
                    "confidence": result["confidence"],
                    "probability": result["probability"]
                })
                logger.info(f"Email {label} testado: {'Spam' if result['is_spam'] else 'Ham'} (conf: {result['confidence']:.2f})")
            except Exception as e:
                logger.error(f"Erro ao testar email: {str(e)}")
    
    # Salvar resultados
    os.makedirs('results', exist_ok=True)
    with open('results/test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Testes do predictor concluídos!")

def test_validation():
    """Testa a validação cruzada do modelo"""
    logger.info("Iniciando validação cruzada...")
    
    try:
        results = cross_validate(
            'data/expanded_spam_ham_dataset.csv',
            n_splits=3,  # Usando menos folds para teste rápido
            batch_size=32,
            learning_rate=0.001
        )
        
        # Verificar métricas
        metrics = results["mean_metrics"]
        logger.info("\nMétricas de validação:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Verificar se as métricas são razoáveis
        assert metrics["accuracy"] > 0.7, "Acurácia muito baixa"
        assert metrics["f1_score"] > 0.7, "F1-score muito baixo"
        
        logger.info("Validação cruzada concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na validação cruzada: {str(e)}")
        raise

def load_model(model_path='best_model.pt'):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model = SpamClassifier()
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict_email(model, tokenizer, device, email_text):
    # Tokenize the email
    encoding = tokenizer(
        email_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move tensors to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = (outputs > 0.5).float()
    
    return prediction.item()

def main():
    """Executa todos os testes"""
    logger.info("Iniciando testes do modelo...")
    
    # Testar predictor
    test_predictor()
    
    # Testar validação
    test_validation()
    
    # Load the model
    print("Loading model...")
    model, tokenizer, device = load_model()
    print("Model loaded successfully!")
    
    # Test emails
    test_emails = [
        {
            "text": "Dear valued customer, your account has been compromised. Please click here to verify your identity: http://suspicious-link.com",
            "expected": "spam"
        },
        {
            "text": "Hi John, I wanted to follow up on our meeting yesterday. The project is progressing well and we should have the first draft ready by next week. Best regards, Sarah",
            "expected": "ham"
        },
        {
            "text": "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize: http://fake-lottery.com",
            "expected": "spam"
        },
        {
            "text": "Meeting reminder: Team sync tomorrow at 10 AM in Conference Room B. Please bring your project updates.",
            "expected": "ham"
        },
        {
            "text": "URGENT: Your account will be suspended in 24 hours unless you verify your information. Click here: http://phishing-site.com",
            "expected": "spam"
        }
    ]
    
    print("\nTesting model with sample emails:")
    print("-" * 80)
    
    for email in test_emails:
        prediction = predict_email(model, tokenizer, device, email["text"])
        result = "spam" if prediction == 1 else "ham"
        print(f"Email: {email['text'][:100]}...")
        print(f"Prediction: {result}")
        print(f"Expected: {email['expected']}")
        print(f"Correct: {'✓' if result == email['expected'] else '✗'}")
        print("-" * 80)
    
    # Test with a few emails from the dataset
    print("\nTesting with real emails from the dataset:")
    print("-" * 80)
    
    df = pd.read_csv('data/expanded_spam_ham_dataset.csv')
    sample_emails = df.sample(n=5)
    
    for _, email in sample_emails.iterrows():
        prediction = predict_email(model, tokenizer, device, email["text"])
        result = "spam" if prediction == 1 else "ham"
        print(f"Email: {email['text'][:100]}...")
        print(f"Prediction: {result}")
        print(f"Actual: {email['label']}")
        print(f"Correct: {'✓' if result == email['label'] else '✗'}")
        print("-" * 80)
    
    logger.info("Todos os testes concluídos com sucesso!")

if __name__ == "__main__":
    main() 