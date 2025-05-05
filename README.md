# Phishing Email Detector

Um detector de emails de phishing usando BERT-tiny para análise de texto, otimizado para baixo consumo de memória e alta performance.

## 🚀 Funcionalidades

- Interface web moderna e responsiva
- Análise de emails em tempo real
- Exemplos de emails para teste
- Detecção de phishing com alta precisão
- Exibição de confiança e probabilidade
- Modo claro/escuro
- Contagem de palavras
- Otimizado para baixo consumo de memória

## 🎯 Como Usar

### Interface Web
1. Acesse a aplicação em `http://localhost:8001` (local) ou sua URL de deploy
2. Você verá uma interface com:
   - Campo de texto para inserir o email
   - Seletor de exemplos pré-carregados
   - Botão de análise
   - Contador de palavras
   - Botão de tema claro/escuro

### Análise de Email
1. **Método 1 - Texto Direto**:
   - Cole o texto do email no campo principal
   - Clique em "Check Email"
   - Aguarde a análise (geralmente 1-2 segundos)
   - Veja o resultado com confiança e probabilidade

2. **Método 2 - Exemplos**:
   - Selecione um exemplo do dropdown
   - O texto será preenchido automaticamente
   - Clique em "Check Email"
   - Analise o resultado

### Resultados
A análise retorna:
- Classificação (Spam/Ham)
- Nível de confiança
- Probabilidade
- Características do texto

### Exemplo de Uso via API
```python
import requests

# URL da API
url = "https://phishing-email-detector-9548433b70c2.herokuapp.com/"

# Email para análise
email_text = """
Subject: Urgent: Your Account Security Alert

Dear Customer,

We have detected unusual activity on your account. Please verify your identity immediately by clicking the link below:

http://suspicious-link.com/verify

Failure to respond within 24 hours will result in account suspension.

Best regards,
Security Team
"""

# Enviar requisição
response = requests.post(url, json={"email_text": email_text})
result = response.json()

print(f"É phishing? {result['is_spam']}")
print(f"Confiança: {result['confidence']:.2%}")
print(f"Probabilidade: {result['probability']:.2%}")
```

### Exemplos de Emails
O sistema inclui exemplos de:
- Emails de phishing
- Emails legítimos
- Emails com características suspeitas
- Emails com urgência falsa

### Dicas de Uso
1. **Para Melhor Precisão**:
   - Inclua o assunto do email
   - Mantenha o texto original
   - Evite textos muito curtos

2. **Interpretando Resultados**:
   - Confiança > 80%: Alta probabilidade de acerto
   - Confiança 50-80%: Requer análise adicional
   - Confiança < 50%: Provavelmente legítimo

3. **Otimização de Performance**:
   - Use o cache de previsões
   - Limite o tamanho do texto
   - Utilize exemplos pré-carregados

## 🏗️ Estrutura do Projeto

```
phishing-email-detector/
├── app/
│   ├── main.py          # API FastAPI e rotas
│   ├── predict.py       # Modelo de predição otimizado
│   ├── bert_classifier.py # Implementação do classificador BERT
│   └── routers/         # Rotas da API
├── static/
│   ├── index.html       # Interface web
│   ├── style.css        # Estilos
│   ├── script.js        # Lógica frontend
│   ├── emails_samples.json # Exemplos de emails
│   └── assets/          # Recursos estáticos
├── models/
│   └── best_model.pt    # Modelo treinado
├── nltk_data/           # Dados do NLTK
├── requirements.txt     # Dependências
└── Procfile            # Configuração de deploy
```

## 📋 Requisitos

- Python 3.11+
- Dependências listadas em `requirements.txt`:
  - FastAPI 0.104.1
  - Uvicorn 0.24.0
  - Transformers 4.35.2
  - PyTorch 2.1.2+cpu
  - NLTK 3.8.1
  - Outras dependências listadas no arquivo

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Audiobf/phishing-email-detector.git
cd phishing-email-detector
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🚀 Execução Local

1. Inicie o servidor:
```bash
uvicorn app.main:app --reload --port 8001
```

2. Acesse a aplicação em:
```
http://localhost:8001
```

## 🌐 Deploy

### Heroku

1. Instale o Heroku CLI
2. Faça login:
```bash
heroku login
```

3. Crie um novo app:
```bash
heroku create seu-app-nome
```

4. Configure as variáveis de ambiente:
```bash
heroku config:set PYTHONPATH=/app
```

5. Faça o deploy:
```bash
git push heroku main
```

### Otimizações de Memória

O projeto implementa várias otimizações para reduzir o uso de memória:
- Singleton pattern para carregamento único do modelo
- Lazy loading do modelo BERT e tokenizer
- Cache LRU para previsões frequentes
- Limite de tamanho do texto de entrada
- Limpeza explícita de memória
- Uso otimizado do token [CLS] para classificação

## 📊 API Endpoints

- `GET /` - Interface web
- `POST /email/analyze` - Analisa um email
- `GET /samples` - Obtém exemplos de emails
- `GET /static/emails_samples.json` - Arquivo de exemplos
- `GET /favicon.ico` - Ícone do site

## 🔍 Detalhes Técnicos

### Modelo BERT
- Usa BERT-tiny para eficiência
- Implementação otimizada para CPU
- Camadas fully connected reduzidas
- Dropout para regularização

### Frontend
- Interface responsiva
- Tema claro/escuro
- Contagem de palavras em tempo real
- Feedback visual de análise
- Exemplos pré-carregados

### Otimizações
- Cache de previsões
- Limpeza de memória automática
- Carregamento sob demanda
- Limite de tamanho de texto
- Singleton pattern

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📝 Licença

MIT

## ⚠️ Limitações

- Tamanho máximo do slug no Heroku: 300MB (soft limit)
- Memória disponível: 512MB (dyno básico)
- Tamanho máximo do texto: 1000 caracteres
- Cache de previsões: últimas 100 previsões