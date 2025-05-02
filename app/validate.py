import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from app.train import EmailClassifier, EmailDataset, load_data
from app.preprocess import TextPreprocessor
import json
import os
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """Calcula várias métricas de avaliação"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

def cross_validate(data_path, n_splits=5, batch_size=32, learning_rate=0.001):
    """Realiza validação cruzada do modelo"""
    logger.info("Iniciando validação cruzada...")
    
    # Carregar dados
    X, y, _ = load_data(data_path)
    input_dim = X.shape[1]
    
    # Configurar K-Fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Métricas
    fold_metrics = []
    fold_losses = []
    
    # Loop de validação cruzada
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        logger.info(f'\nFold {fold + 1}/{n_splits}')
        
        # Dividir dados
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Criar datasets e dataloaders
        train_dataset = EmailDataset(X_train, y_train)
        val_dataset = EmailDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size
        )
        
        # Inicializar modelo
        model = EmailClassifier(input_dim)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Treinar modelo
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(20):  # Número máximo de épocas
            # Treinamento
            model.train()
            train_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validação
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    val_loss += loss.item()
                    
                    preds = (outputs >= 0.5).float()
                    all_preds.extend(preds.squeeze().tolist())
                    all_labels.extend(labels.tolist())
            
            # Calcular métricas
            metrics = calculate_metrics(all_labels, all_preds)
            val_accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
            
            logger.info(f'Epoch {epoch+1}/20:')
            logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}')
            logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}')
            logger.info(f'Val Accuracy: {val_accuracy:.2f}%')
            logger.info(f'Precision: {metrics["precision"]:.4f}')
            logger.info(f'Recall: {metrics["recall"]:.4f}')
            logger.info(f'F1 Score: {metrics["f1_score"]:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Salvar métricas do fold
        fold_metrics.append(metrics)
        fold_losses.append(val_loss/len(val_loader))
    
    # Calcular métricas finais
    mean_metrics = {
        'accuracy': np.mean([(m['true_positives'] + m['true_negatives']) / 
                           (m['true_positives'] + m['true_negatives'] + 
                            m['false_positives'] + m['false_negatives']) 
                           for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in fold_metrics]),
        'mean_loss': np.mean(fold_losses)
    }
    
    std_metrics = {
        'accuracy': np.std([(m['true_positives'] + m['true_negatives']) / 
                          (m['true_positives'] + m['true_negatives'] + 
                           m['false_positives'] + m['false_negatives']) 
                          for m in fold_metrics]),
        'precision': np.std([m['precision'] for m in fold_metrics]),
        'recall': np.std([m['recall'] for m in fold_metrics]),
        'f1_score': np.std([m['f1_score'] for m in fold_metrics]),
        'mean_loss': np.std(fold_losses)
    }
    
    logger.info('\nCross-validation Results:')
    for metric in mean_metrics:
        logger.info(f'{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}')
    
    # Salvar resultados
    results = {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'fold_metrics': fold_metrics,
        'fold_losses': [float(loss) for loss in fold_losses]
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/cross_validation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == '__main__':
    data_path = 'data/expanded_spam_ham_dataset.csv'
    results = cross_validate(data_path)
    logger.info("Validação cruzada concluída com sucesso!") 