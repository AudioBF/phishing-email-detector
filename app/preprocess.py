import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import email
from email import policy
from email.parser import BytesParser
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure NLTK data directory
try:
    nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_dir)
except Exception as e:
    logger.error(f"Error initializing NLTK: {e}")
    raise

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Lista de palavras-chave comuns em phishing (atualizada)
        self.spam_keywords = {
            'urgent', 'free', 'winner', 'lottery', 'click', 'password', 'account',
            'verify', 'security', 'update', 'limited', 'offer', 'congratulations',
            'prize', 'win', 'selected', 'exclusive', 'guaranteed', 'risk-free',
            'special', 'promotion', 'discount', 'save', 'money', 'cash', 'bonus',
            'million', 'dollars', 'investment', 'opportunity', 'claim', 'suspicious',
            'hack', 'compromise', 'breach', 'alert', 'warning', 'immediate', 'action',
            'required', 'suspended', 'locked', 'unlock', 'verify', 'confirm', 'secure',
            'protect', 'fraud', 'scam', 'phishing', 'malware', 'virus', 'threat',
            'login', 'credentials', 'password', 'username', 'account', 'verify',
            'confirm', 'validate', 'authenticate', 'access', 'login', 'signin',
            'sign-in', 'sign_in', 'log-in', 'log_in', 'password', 'reset', 'change',
            'update', 'security', 'alert', 'warning', 'notice', 'important',
            'action', 'required', 'immediately', 'urgent', 'critical', 'emergency',
            'suspicious', 'unusual', 'activity', 'detected', 'compromised', 'hacked',
            'breach', 'leak', 'exposed', 'stolen', 'unauthorized', 'access',
            'blocked', 'locked', 'suspended', 'terminated', 'closed', 'expired',
            'verify', 'confirm', 'validate', 'authenticate', 'identity', 'personal',
            'information', 'data', 'privacy', 'security', 'protection', 'safe',
            'secure', 'trust', 'reliable', 'legitimate', 'official', 'authorized',
            'certified', 'verified', 'authentic', 'genuine', 'real', 'true',
            'valid', 'legal', 'lawful', 'legitimate', 'official', 'authorized',
            'certified', 'verified', 'authentic', 'genuine', 'real', 'true',
            'valid', 'legal', 'lawful'
        }
        
        # Lista de palavras-chave comuns em emails legítimos (atualizada)
        self.ham_keywords = {
            'meeting', 'project', 'team', 'report', 'discuss', 'schedule',
            'agenda', 'minutes', 'follow-up', 'deadline', 'review', 'feedback',
            'update', 'status', 'progress', 'collaboration', 'coordinate',
            'department', 'manager', 'colleague', 'client', 'customer', 'invoice',
            'payment', 'receipt', 'order', 'confirmation', 'appointment', 'booking',
            'reservation', 'ticket', 'flight', 'hotel', 'conference', 'seminar',
            'workshop', 'training', 'document', 'file', 'attachment', 'contract',
            'agreement', 'proposal', 'presentation', 'analysis', 'data', 'report',
            'summary', 'conclusion', 'recommendation', 'suggestion', 'feedback',
            'question', 'answer', 'clarification', 'explanation', 'information',
            'details', 'specifications', 'requirements', 'deadline', 'timeline',
            'milestone', 'deliverable', 'resource', 'budget', 'cost', 'expense',
            'revenue', 'profit', 'loss', 'financial', 'accounting', 'audit',
            'compliance', 'regulation', 'policy', 'procedure', 'guideline',
            'standard', 'quality', 'performance', 'efficiency', 'productivity',
            'improvement', 'optimization', 'enhancement', 'development', 'growth',
            'expansion', 'strategy', 'planning', 'execution', 'implementation',
            'monitoring', 'evaluation', 'assessment', 'analysis', 'research',
            'study', 'survey', 'interview', 'discussion', 'negotiation',
            'agreement', 'contract', 'partnership', 'collaboration', 'alliance',
            'network', 'community', 'organization', 'company', 'business',
            'enterprise', 'industry', 'sector', 'market', 'competition',
            'customer', 'client', 'user', 'consumer', 'stakeholder', 'partner',
            'supplier', 'vendor', 'contractor', 'consultant', 'advisor',
            'expert', 'specialist', 'professional', 'employee', 'staff',
            'worker', 'manager', 'director', 'executive', 'officer',
            'president', 'chairman', 'board', 'committee', 'council',
            'department', 'division', 'unit', 'team', 'group', 'project',
            'program', 'initiative', 'campaign', 'event', 'activity',
            'operation', 'process', 'procedure', 'method', 'approach',
            'technique', 'tool', 'resource', 'material', 'equipment',
            'facility', 'infrastructure', 'system', 'platform', 'software',
            'application', 'program', 'code', 'database', 'server', 'network',
            'security', 'privacy', 'confidentiality', 'integrity', 'reliability',
            'availability', 'performance', 'efficiency', 'quality', 'standard',
            'compliance', 'regulation', 'policy', 'law', 'rule', 'guideline'
        }

    def preprocess_text(self, text):
        """Pré-processamento básico do texto"""
        # Converter para minúsculas
        text = text.lower()
        
        # Extrair URLs antes de removê-las
        urls = re.findall(r'http\S+|www\S+|https\S+', text, flags=re.MULTILINE)
        url_count = len(urls)
        
        # Extrair emails antes de removê-los
        emails = re.findall(r'\S+@\S+', text)
        email_count = len(emails)
        
        # Remover URLs e emails
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Preservar números e caracteres especiais importantes
        text = re.sub(r'[^\w\s\d\-_@#$%&*]', ' ', text)
        
        # Remover espaços extras
        text = ' '.join(text.split())
        
        return text, url_count, email_count

    def extract_features(self, text):
        """Extrai features adicionais do texto"""
        features = {}
        
        # Pré-processamento com preservação de informações importantes
        processed_text, url_count, email_count = self.preprocess_text(text)
        
        # Features básicas
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Tokenização e stemming
        tokens = word_tokenize(processed_text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Features de phishing
        spam_count = sum(1 for word in stemmed_tokens if word in self.spam_keywords)
        features['spam_keywords'] = spam_count
        features['spam_keyword_ratio'] = spam_count / features['word_count'] if features['word_count'] > 0 else 0
        
        # Features de ham
        ham_count = sum(1 for word in stemmed_tokens if word in self.ham_keywords)
        features['ham_keywords'] = ham_count
        features['ham_keyword_ratio'] = ham_count / features['word_count'] if features['word_count'] > 0 else 0
        
        # Features de estrutura
        features['has_links'] = url_count
        features['has_emails'] = email_count
        features['has_caps'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Features de formatação
        features['has_attachments'] = 1 if 'attachment' in text.lower() else 0
        features['has_signature'] = 1 if any(sig in text.lower() for sig in ['regards', 'best regards', 'sincerely', 'thank you', 'thanks']) else 0
        
        # Features adicionais
        features['has_subject'] = 1 if 'subject:' in text.lower() else 0
        features['has_recipient'] = 1 if 'dear' in text.lower() or 'hi' in text.lower() or 'hello' in text.lower() else 0
        features['has_date'] = 1 if any(date in text.lower() for date in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']) else 0
        features['has_time'] = 1 if re.search(r'\d{1,2}:\d{2}\s*(am|pm)?', text.lower()) else 0
        features['has_location'] = 1 if any(loc in text.lower() for loc in ['room', 'floor', 'building', 'address', 'location', 'venue']) else 0
        
        # Novas features para melhorar a detecção
        features['has_company_info'] = 1 if any(info in text.lower() for info in ['company', 'organization', 'department', 'team']) else 0
        features['has_project_info'] = 1 if any(info in text.lower() for info in ['project', 'task', 'assignment', 'deadline']) else 0
        features['has_meeting_info'] = 1 if any(info in text.lower() for info in ['meeting', 'agenda', 'minutes', 'schedule']) else 0
        features['has_business_terms'] = 1 if any(term in text.lower() for term in ['invoice', 'payment', 'contract', 'agreement']) else 0
        
        # Features de contexto
        features['has_urgency'] = 1 if any(word in text.lower() for word in ['urgent', 'immediate', 'asap', 'right away']) else 0
        features['has_threat'] = 1 if any(word in text.lower() for word in ['suspended', 'blocked', 'terminated', 'closed']) else 0
        features['has_reward'] = 1 if any(word in text.lower() for word in ['prize', 'win', 'bonus', 'reward']) else 0
        features['has_security'] = 1 if any(word in text.lower() for word in ['security', 'verify', 'authenticate', 'password']) else 0
        
        # Balanceamento de features
        features['spam_ham_ratio'] = features['spam_keyword_ratio'] / (features['ham_keyword_ratio'] + 0.0001)  # Evita divisão por zero
        features['keyword_balance'] = features['ham_keywords'] - features['spam_keywords']
        
        # Features de confiança
        features['business_context_score'] = sum([
            features['has_company_info'],
            features['has_project_info'],
            features['has_meeting_info'],
            features['has_business_terms']
        ]) / 4.0
        
        features['spam_context_score'] = sum([
            features['has_urgency'],
            features['has_threat'],
            features['has_reward'],
            features['has_security']
        ]) / 4.0
        
        return features

    def process_email(self, email_text):
        """Processa um email completo e retorna o texto pré-processado e features"""
        # Pré-processamento básico
        processed_text, _, _ = self.preprocess_text(email_text)
        
        # Extração de features
        features = self.extract_features(email_text)
        
        return processed_text, features 