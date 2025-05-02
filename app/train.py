import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import argparse
import os
import json
from pathlib import Path
from app.preprocess import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EmailClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EmailClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

def load_data(file_path):
    """Load and preprocess the data"""
    logger.info("Loading data...")
    
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Check columns
    logger.info("Available columns: %s", df.columns.tolist())
    
    # Assuming we have 'text' and 'label' or similar columns
    text_column = 'text' if 'text' in df.columns else 'email'
    label_column = 'label' if 'label' in df.columns else 'spam'
    
    preprocessor = TextPreprocessor()
    X_text = []
    X_features = []
    y = []
    
    # Process each email
    for _, row in df.iterrows():
        processed_text, features = preprocessor.process_email(str(row[text_column]))
        X_text.append(processed_text)
        X_features.append(list(features.values()))
        y.append(1 if str(row[label_column]).lower() == 'spam' else 0)
    
    # Convert features to numpy array
    X_features = np.array(X_features)
    
    # Scale features
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    
    # Log dataset statistics
    logger.info("Total emails: %d", len(y))
    logger.info("Spam emails: %d", sum(y))
    logger.info("Ham emails: %d", len(y) - sum(y))
    
    return X_features, np.array(y), scaler

def train_model(X, y, input_dim, epochs=100, batch_size=32, learning_rate=0.001):
    """Train the model"""
    logger.info("Training model...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = EmailDataset(X_train, y_train)
    val_dataset = EmailDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = EmailClassifier(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Define model paths
    best_model_path = os.path.join('models', 'email_classifier.pth')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch.unsqueeze(1)).item()
                predicted = (y_pred > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted.squeeze() == y_batch).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Accuracy: {accuracy:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train email classifier model')
    parser.add_argument('--data', type=str, default='data/spam_ham_dataset.csv',
                      help='Path to the dataset CSV file')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate for optimizer')
    args = parser.parse_args()

    # Clean up old model files
    model_files = ['best_model.pth', 'best_model.pt', 'spam_model.pt']
    for file in model_files:
        file_path = os.path.join('models', file)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed old model file: {file_path}")

    # Load data
    X, y, scaler = load_data(args.data)
    input_dim = X.shape[1]
    
    # Train model
    model = train_model(X, y, input_dim, epochs=args.epochs, 
                       batch_size=args.batch_size, learning_rate=args.learning_rate)
    
    # Save scaler
    scaler_path = os.path.join('models', 'scaler.pkl')
    torch.save(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save training statistics
    stats = {
        'input_dim': input_dim,
        'total_samples': len(y),
        'spam_samples': int(sum(y)),
        'ham_samples': int(len(y) - sum(y))
    }
    stats_path = os.path.join('models', 'training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Saved training statistics to {stats_path}")

if __name__ == "__main__":
    main() 