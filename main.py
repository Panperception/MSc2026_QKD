import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from src.dataset import get_qkd_dataloaders
from src.model import QKDGatekeeperLSTM

# parameters and configuration 
WINDOW_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

# use gpu if available if not use cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, train_loader, criterion, optimiser, epochs, device):
    # training loop for the LSTM model

    model.to(device)
    
    train_losses = []
    
    for i in range(epochs):
        #set model to train
        model.train()
        
        running_loss = 0.0
        # for each batch of windows in the dataset
        # move tensors and targets to device and predict probabilities for current batch
        # then calaculate errors
        # then do backward pass and calculate gradient of loss
        # then update model weights
        # and add upp total weight
        for X_batch,y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions,y_batch)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        # calculate store and loss for epoch
        train_losses.append(running_loss/len(train_loader))
        print(f"Epoch [{i+1}/{epochs} - Loss: {running_loss:.5f}]")
    
    return train_losses


def evaluate_model(model,test_loader,device):
    """
        evaluates trained model on test telemtry data
    """
    model.to(device)

    #set model to evaluation mode
    model.eval()
    probs = []
    targets = []

    with torch.no_grad():
        for X_batch,y_batch in test_loader:
            # inputs into device
            X_device = X_batch.to(device)

            # calculate anomaly probabilities
            batch_probs = model(X_device)

            # probs back to cpu and convert to numpy
            probs.extend(batch_probs.cpu().numpy())
            targets.extend(y_batch.numpy())
            
    probs = np.array(probs)
    targets = np.array(targets)
    # converting probabilites into binary classification 
    # 0 = no attack, 1 = attack
    predictions = (probs >= 0.5).astype(int)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS (TEST SET)")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=["Normal", "Attack"]))
    
    auc_score = roc_auc_score(targets, probs)
    print(f"ROC-AUC Score: {auc_score:.4f}\n")
    
    return targets, probs, predictions
            

def plot_metrics(train_losses, targets, probs):
    os.makedirs('plots', exist_ok=True)
    
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', color='crimson', label='BCE Loss')
    plt.title('Training Loss Curve', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/training_loss_curve.png', dpi=300)
    plt.close()
    
    
    fpr, tpr, _ = roc_curve(targets, probs)
    auc_val = roc_auc_score(targets, probs)
    
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='navy', lw=2, label=f'LSTM Gatekeeper (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png', dpi=300)
    plt.close()
    
    print("Plots saved to 'plots/training_loss_curve.png' and 'plots/roc_curve.png'")    
        
        

if __name__ == '__main__':
    print(f"Using Compute Device: {DEVICE}")
    
    
    train_loader, test_loader, scaler = get_qkd_dataloaders(window_size=WINDOW_SIZE, batch_size=BATCH_SIZE)
    
    
    model = QKDGatekeeperLSTM(input_size=3, hidden_dim=64, num_layers=2)
    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    
    print("\nStarting Training...")
    train_losses = train_model(model, train_loader, criterion, optimiser, EPOCHS, DEVICE)
    
    
    targets, probs, preds = evaluate_model(model, test_loader, DEVICE)
    
    
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/qkd_gatekeeper_lstm.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model checkpoint saved to {model_save_path}")
    

    plot_metrics(train_losses, targets, probs)