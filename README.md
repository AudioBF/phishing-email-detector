# Phishing Email Detector

Um detector de emails de phishing usando BERT-tiny para análise de texto.

## Funcionalidades

- Interface web amigável
- Análise de emails em tempo real
- Exemplos de emails para teste
- Detecção de phishing com alta precisão
- Exibição de confiança e probabilidade

## Estrutura do Projeto

```
phishing-email-detector/
├── app/
│   ├── main.py          # API FastAPI
│   └── predict.py       # Modelo de predição
├── static/
│   ├── index.html       # Interface web
│   ├── style.css        # Estilos
│   ├── script.js        # Lógica frontend
│   └── emails_samples.json # Exemplos de emails
├── models/
│   └── best_model.pt    # Modelo treinado
├── requirements.txt     # Dependências
└── Procfile            # Configuração de deploy
```

## Requisitos

- Python 3.8+
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/phishing-email-detector.git
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

## Execução Local

1. Inicie o servidor:
```bash
uvicorn app.main:app --reload --port 8001
```

2. Acesse a aplicação em:
```
http://localhost:8001
```

## Deploy

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

### Outras Plataformas

O projeto pode ser deployado em qualquer plataforma que suporte Python e FastAPI, como:
- Railway
- Render
- DigitalOcean
- AWS Elastic Beanstalk

Certifique-se de configurar:
- Variável de ambiente `PORT`
- Dependências Python
- Arquivos estáticos

## Métricas

O modelo fornece as seguintes métricas:
- Acurácia
- Precisão
- Recall
- F1-score
- Matriz de confusão

## API Endpoints

- `POST /analyze` - Analisa um email
- `POST /validate` - Executa validação cruzada
- `GET /metrics` - Obtém métricas do modelo
- `GET /samples` - Obtém exemplos de emails

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

MIT