import torch
import numpy as np

def load_model(model_class, weights_path, *model_args, device='cpu'):
    model = model_class(*model_args)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, data_loader, device):
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(actuals)
