import requests
import json

# URL da API
url = "http://localhost:8001/analyze"

# Email de teste
test_email = {
    "text": "Dear valued customer,\n\nWe have noticed suspicious activity on your account. To protect your account, please click on the link below to verify your information:\n\nhttp://fake-bank-verify.com/account\n\nIf you do not verify your account within 24 hours, your account will be suspended.\n\nBest regards,\nBank Security Team"
}

# Enviar requisição
response = requests.post(url, json=test_email)

# Imprimir resultado
print("Status Code:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2)) 