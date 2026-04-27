import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def train_model(model, train_loader, val_loader, epochs, lr, device, pos_weight, weight_decay):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y, batch_weights in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_weights in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                val_loss += criterion(logits, batch_y).item()

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        # Tính toán Metrics nâng cao
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        val_prec = precision_score(all_targets, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "f"Val Loss: {val_loss/len(val_loader):.4f} | Val F1: {val_f1:.4f} | Val Precision: {val_prec:.4f}")