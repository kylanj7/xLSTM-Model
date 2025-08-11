import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
from datetime import datetime
from config import MODELS_DIR, get_timestamp  # import config constants

def train_model(model, train_loader, val_loader, device, learning_rate=0.001, epochs=1, patience=15):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_loss = float('inf')
    counter = 0
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                loss = criterion(model(batch_x), batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            timestamp = get_timestamp()
            best_model_path = os.path.join(MODELS_DIR, f'best_model_{timestamp}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to: {best_model_path}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights before returning
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))

    return model, train_losses, val_losses

