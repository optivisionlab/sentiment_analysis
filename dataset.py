import torch
import numpy as np
from torch.utils.data import Dataset
from collections import Counter

class StockNewsDataset(Dataset):
    def __init__(self, df, embeddings, seq_length=5):
        self.X, self.y, self.weights = [], [], []
        
        binary_targets = df['Binary_Target'].values
        target_counts = Counter(binary_targets)
        pos_weight_val = target_counts.get(0, 1) / target_counts.get(1, 1)
        weight_dict = {0: 1.0, 1: pos_weight_val}

        grouped = df.groupby('StockCode')

        for stock, group in grouped:
            indices = group.index.values
            if len(indices) >= seq_length:
                for i in range(len(indices) - seq_length + 1):
                    window_indices = indices[i : i + seq_length]
                    window_embeddings = embeddings[window_indices]
                    
                    target_label = df.loc[indices[i + seq_length - 1], 'Binary_Target']
                    current_weight = weight_dict.get(target_label, 1.0)

                    self.X.append(window_embeddings)
                    self.y.append(float(target_label))
                    self.weights.append(current_weight)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)
        self.weights = torch.tensor(np.array(self.weights), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]