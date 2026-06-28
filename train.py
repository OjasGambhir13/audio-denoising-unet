
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from dataset import get_dataloaders
from model import AudioUNet

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AudioUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_loader, val_loader = get_dataloaders('./datasets/clean', './datasets/noise', batch_size=8)
    
    epochs = 50
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for noisy_mag, clean_mag in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            noisy_mag, clean_mag = noisy_mag.to(device), clean_mag.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy_mag)
            loss = criterion(outputs, clean_mag)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for noisy_mag, clean_mag in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                noisy_mag, clean_mag = noisy_mag.to(device), clean_mag.to(device)
                outputs = model(noisy_mag)
                loss = criterion(outputs, clean_mag)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'models/best_unet.pth')
            print(">> Best model saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('outputs/loss_curve.png')
    plt.close()

if __name__ == '__main__':
    train()