import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from src.config import Config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_logs(logs):
    df_logs = pd.DataFrame(logs)
    os.makedirs(Config.LOG_FILE, exist_ok = True)
    df_logs.to_excel(os.path.join(Config.LOG_FILE, "training_logs.xlsx"), index=False)
    print(f"Saved training logs to {Config.LOG_FILE}")
    return df_logs

def plot_metrics(df_logs, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    metrics = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Validation Loss"),
        ("accuracy", "Validation Accuracy"),
        ("f1", "Validation F1-score")
    ]
    for y_col, title in metrics:
        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=df_logs,
            x="epoch",
            y=y_col,
            hue="fold",
            marker="o",
            linewidth=2
        )
        plt.title(f"{title} per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend(title="Fold")
        plt.tight_layout()
        filename = f"{y_col}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()