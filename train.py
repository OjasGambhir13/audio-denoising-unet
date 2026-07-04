import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from dataset import get_dataloaders
from model import AudioUNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-5  # Lower learning rate for fine-tuning
    patience = 5          # Early stopping patience limit

    clean_dir = './datasets/clean'
    noise_dir = './datasets/noise'
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_unet.pth'

    model = AudioUNet().to(device)

    if os.path.exists(model_path):
        print(f"🧠 Loading existing AI brain from {model_path} to enhance it...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print("⚠️ No existing model found. Training from scratch.")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler = GradScaler() # Speeds up training on RTX GPUs

    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(clean_dir, noise_dir, batch_size=batch_size)

    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for noisy_mag, clean_mag in train_bar:
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            optimizer.zero_grad()

            with autocast():
                denoised_mag = model(noisy_mag)
                loss = criterion(denoised_mag, clean_mag)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for noisy_mag, clean_mag in val_bar:
                noisy_mag = noisy_mag.to(device)
                clean_mag = clean_mag.to(device)

                with autocast():
                    denoised_mag = model(noisy_mag)
                    loss = criterion(denoised_mag, clean_mag)

                val_loss += loss.item()
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(">> Enhanced model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. Model is fully optimized!")
                break

if __name__ == '__main__':
    main()