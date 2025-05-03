# app/predict.py
import torch
import logging
from transformers import BertTokenizer
import sys
import os
from pathlib import Path

# Adiciona o diretório atual ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.bert_classifier import SpamClassifier

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailPredictor:
    def __init__(self, model_path='best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Carregar o tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
            logger.info("Tokenizer carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar tokenizer: {str(e)}")
            raise
        
        # Carregar o modelo
        try:
            if os.path.exists(model_path):
                self.model = SpamClassifier()
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Modelo BERT carregado com sucesso")
            else:
                # Se o modelo não existir, usar o modelo pré-treinado
                self.model = SpamClassifier()
                logger.info("Usando modelo BERT pré-treinado")
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def predict(self, email_text):
        try:
            # Tokenizar o email
            encoding = self.tokenizer(
                email_text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Mover tensores para o device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Fazer predição
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probability = outputs.item()
            
            # Usar threshold de 0.5 e confiança baseada na probabilidade
            is_spam = probability > 0.5
            confidence = abs(probability - 0.5) * 2  # Normaliza para 0-1
            
            # Log para debug
            logger.info(f"Probabilidade: {probability}")
            logger.info(f"Classificação: {'Spam' if is_spam else 'Ham'}")
            logger.info(f"Confiança: {confidence}")
            
            return {
                'is_spam': is_spam,
                'confidence': confidence,
                'probability': probability,
                'features': {
                    'text_length': len(email_text),
                    'has_links': 'http' in email_text.lower() or 'www' in email_text.lower(),
                    'has_urgency': any(word in email_text.lower() for word in ['urgent', 'immediately', 'now', 'asap']),
                    'has_money': any(word in email_text.lower() for word in ['$', 'money', 'cash', 'prize', 'win']),
                    'has_suspicious': any(word in email_text.lower() for word in ['verify', 'account', 'password', 'click'])
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            raise

def predict_email(email_text: str) -> dict:
    """
    Predict whether an email is spam or not using BERT.
    
    Args:
        email_text (str): The email text to analyze
        
    Returns:
        dict: Dictionary containing prediction results
    """
    try:
        # Inicializar o predictor
        predictor = EmailPredictor()
        
        # Fazer a predição
        result = predictor.predict(email_text)
        
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
