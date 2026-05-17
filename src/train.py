import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from src.classifier import SentimentModel
from src.config import Config

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_text, x_stock, x_type, y in loader:
        x_text = x_text.to(device)
        x_stock = x_stock.to(device)
        x_type = x_type.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x_text, x_stock, x_type)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return logits, total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds, labels = [], []

    for x_text, x_stock, x_type, y in loader:
        x_text = x_text.to(device)
        x_stock = x_stock.to(device)
        x_type = x_type.to(device)
        y = y.to(device).unsqueeze(1)

        logits = model(x_text, x_stock, x_type)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long().cpu().numpy().flatten()

        preds.extend(pred.tolist())
        labels.extend(y.long().cpu().numpy().flatten().tolist())

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
    precision = precision_score(labels, preds, zero_division=0) if len(labels) > 0 else 0.0
    recall = recall_score(labels, preds, zero_division=0) if len(labels) > 0 else 0.0
    f1 = f1_score(labels, preds, zero_division=0) if len(labels) > 0 else 0.0

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, acc, precision, recall, f1


def fit_model(train_dataset, val_dataset, num_stocks, num_types):
    set_seed(Config.RANDOM_STATE)
    device = Config.DEVICE

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=Config.DROP_LAST
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False
    )

    model = SentimentModel(num_stocks=num_stocks, num_types=num_types).to(device)

    pos_weight = _get_pos_weight(train_dataset.y).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )
    all_logs = []
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, acc, precision, recall, f1 = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        all_logs.append({
            "fold": 0,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        os.makedirs(Config.MODEL_PATH, exist_ok = True)
        torch.save(model.state_dict(), os.path.join(Config.MODEL_PATH, "epoch_{epoch}.pt"))
    return model, all_logsnfig
from src.utils import set_seed
import os


def _get_pos_weight(y_tensor: torch.Tensor):
    y_np = y_tensor.detach().cpu().numpy().astype(int)
    pos = (y_np == 1).sum()
    neg = (y_np == 0).sum()
    if pos == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg / pos, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_text, x_stock, x_type, y in loader:
        x_text = x_text.to(device)
        x_stock = x_stock.to(device)
        x_type = x_type.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x_text, x_stock, x_type)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds, labels = [], []

    for x_text, x_stock, x_type, y in loader:
        x_text = x_text.to(device)
        x_stock = x_stock.to(device)
        x_type = x_type.to(device)
        y = y.to(device).unsqueeze(1)

        logits = model(x_text, x_stock, x_type)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long().cpu().numpy().flatten()

        preds.extend(pred.tolist())
        labels.extend(y.long().cpu().numpy().flatten().tolist())

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
    precision = precision_score(labels, preds, zero_division=0) if len(labels) > 0 else 0.0
    recall = recall_score(labels, preds, zero_division=0) if len(labels) > 0 else 0.0
    f1 = f1_score(labels, preds, zero_division=0) if len(labels) > 0 else 0.0

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, acc, precision, recall, f1


def fit_model(train_dataset, val_dataset, num_stocks, num_types):
    set_seed(Config.RANDOM_STATE)
    device = Config.DEVICE

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=Config.DROP_LAST
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False
    )

    model = SentimentModel(num_stocks=num_stocks, num_types=num_types).to(device)

    pos_weight = _get_pos_weight(train_dataset.y).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )
    all_logs = []
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, acc, precision, recall, f1 = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        all_logs.append({
            "fold": 0,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        os.makedirs(Config.MODEL_PATH, exist_ok = True)
        torch.save(model.state_dict(), os.path.join(Config.MODEL_PATH, f"epoch_{epoch}.pt"))
    return model, all_logs


@torch.no_grad()
def test_model(model, test_dataset):
    device = Config.DEVICE
    criterion = nn.BCEWithLogitsLoss()

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False
    )

    test_loss, acc, precision, recall, f1 = evaluate(model, test_loader, criterion, device)

    return {
        "loss": test_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }