import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import gc

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class SpamClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(SpamClassifier, self).__init__()
        
        # Carregar o modelo BERT pré-treinado
        self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny')
        
        # Camadas fully connected
        self.fc1 = nn.Linear(128, 64)  # 128 é o tamanho do embedding do BERT-tiny
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)
        
        # Forçar limpeza de memória
        gc.collect()
        torch.cuda.empty_cache()
    
    def forward(self, input_ids, attention_mask):
        # Obter embeddings do BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Usar apenas o embedding do token [CLS]
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Aplicar dropout
        pooled_output = self.dropout(pooled_output)
        
        # Passar pelas camadas fully connected
        x = torch.relu(self.fc1(pooled_output))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Limpar memória
        del outputs, pooled_output
        gc.collect()
        torch.cuda.empty_cache()
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=3):
    best_val_loss = float('inf')
    best_model_path = 'best_model.pt'
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print('Model saved!')
        
        print(classification_report(all_labels, all_preds))
        print('-' * 50)

def main():
    # Load and preprocess data
    df = pd.read_csv('data/expanded_spam_ham_dataset.csv')
    texts = df['text'].values
    # Convert labels to numeric (0 for ham, 1 for spam)
    labels = df['label'].map({'ham': 0, 'spam': 1}).values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model = SpamClassifier()
    
    # Create datasets
    train_dataset = EmailDataset(X_train, y_train, tokenizer)
    val_dataset = EmailDataset(X_val, y_val, tokenizer)
    test_dataset = EmailDataset(X_test, y_test, tokenizer)
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print('Test Set Results:')
    print(classification_report(all_labels, all_preds))

if __name__ == '__main__':
    main() 