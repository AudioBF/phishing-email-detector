# app/predict.py
import torch
import logging
from transformers import BertTokenizer
import sys
import os
from pathlib import Path
from functools import lru_cache
import gc

# Adiciona o diretório atual ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.bert_classifier import SpamClassifier

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailPredictor:
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, model_path='best_model.pt'):
        if cls._instance is None:
            cls._instance = super(EmailPredictor, cls).__new__(cls)
            cls._instance._initialize(model_path)
        return cls._instance
    
    def _initialize(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.max_length = 128  # Limitar o tamanho máximo do texto
        
        # Forçar limpeza de memória
        gc.collect()
        torch.cuda.empty_cache()
    
    def _load_tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
                logger.info("Tokenizer carregado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao carregar tokenizer: {str(e)}")
                raise
        return self._tokenizer
    
    def _load_model(self):
        if self._model is None:
            try:
                if os.path.exists(self.model_path):
                    model = SpamClassifier()
                    model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    logger.info("Modelo BERT carregado com sucesso")
                else:
                    model = SpamClassifier()
                    logger.info("Usando modelo BERT pré-treinado")
                
                model.to(self.device)
                model.eval()
                self._model = model
            except Exception as e:
                logger.error(f"Erro ao carregar modelo: {str(e)}")
                raise
        return self._model
    
    @lru_cache(maxsize=100)  # Cache para as últimas 100 previsões
    def predict(self, email_text):
        try:
            # Limitar o tamanho do texto
            if len(email_text) > 1000:  # Limite arbitrário, ajuste conforme necessário
                email_text = email_text[:1000]
            
            # Carregar tokenizer e modelo sob demanda
            tokenizer = self._load_tokenizer()
            model = self._load_model()
            
            # Tokenizar o email
            encoding = tokenizer(
                email_text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Mover para o dispositivo
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Fazer a previsão
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probabilities = torch.sigmoid(outputs)
                prediction = (probabilities > 0.5).float()
            
            # Limpar memória
            del input_ids, attention_mask, outputs
            gc.collect()
            torch.cuda.empty_cache()
            
            return {
                'is_spam': bool(prediction.item()),
                'confidence': float(probabilities.item()),
                'probability': float(probabilities.item()),
                'features': {
                    'text_length': len(email_text),
                    'tokens_count': len(encoding['input_ids'][0])
                }
            }
        except Exception as e:
            logger.error(f"Erro ao fazer previsão: {str(e)}")
            raise

# Função de conveniência para uso externo
def predict_email(email_text: str) -> dict:
    predictor = EmailPredictor()
    return predictor.predict(email_text)
