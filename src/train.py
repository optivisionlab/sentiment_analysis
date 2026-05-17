import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from src.classifier import SentimentModel
from src.dataloader import StockDataset
from src.config import Config


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x_text, x_stock, x_type, x_days,  y in loader:
        x_text, x_stock, x_type, x_days = x_text.to(Config.DEVICE), x_stock.to(Config.DEVICE), x_type.to(Config.DEVICE), x_days.to(Config.DEVICE)
        y = y.to(Config.DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(x_text, x_stock, x_type, x_days)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, labels = [], []

    with torch.no_grad():
        for x_text, x_stock, x_type, x_days,  y in loader:
            x_text, x_stock, x_type, x_days = x_text.to(Config.DEVICE), x_stock.to(Config.DEVICE), x_type.to(Config.DEVICE), x_days.to(Config.DEVICE)
            y = y.to(Config.DEVICE).unsqueeze(1)

            logits = model(x_text, x_stock, x_type, x_days)
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).int().cpu().numpy()

            preds.extend(pred.flatten())
            labels.extend(y.cpu().numpy().flatten())

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return total_loss, acc, precision, recall, f1

def train_kfold(X_text, X_stock, X_type, X_day, Y, num_stocks, num_types):
    skf = StratifiedKFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=42)
    all_logs = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, Y)):
        print(f"\n===== Fold {fold} =====")

        train_dataset = StockDataset(X_text[train_idx], X_stock[train_idx], X_type[train_idx], X_day[train_idx], Y[train_idx])
        val_dataset = StockDataset(X_text[val_idx], X_stock[val_idx], X_type[val_idx], X_day[train_idx], Y[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

        model = SentimentModel(
            num_stocks=num_stocks,
            num_types=num_types,
        ).to(Config.DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        for epoch in range(Config.EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, acc, precision, recall, f1 = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f}")
            all_logs.append({
                "fold": fold, "epoch": epoch + 1, "train_loss": train_loss, 
                "val_loss": val_loss, "accuracy": acc, "precision": precision, 
                "recall": recall, "f1": f1
            })
        fold_scores.append(acc)
    print(f"\nCV Mean Accuracy: {np.mean(fold_scores):.4f}")
    return model, all_logs


def test(model, X_text, X_stock, X_type, X_days, Y_target):
    test_dataset = StockDataset(
        X_text=X_text, 
        X_stock=X_stock, 
        X_type=X_type, 
        X_days=X_days, 
        y=Y_target
    )
    criterion = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    total_loss, acc, precision, recall, f1 = evaluate(model, test_loader, criterion)
    result = {
        "Total_Loss": total_loss,
        "Acc": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    return result