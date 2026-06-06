import torch
import torch.nn as nn
from src.config import Config


class SentimentModel(nn.Module):
    def __init__( self,num_stocks, num_types):
        super().__init__()
        self.phobert_proj = nn.Sequential(
            nn.Linear(Config.PHOBERT_DIM, Config.PROJ_DIM),
            nn.LayerNorm(Config.PROJ_DIM),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.stock_embedding = nn.Embedding(
            num_stocks,
            Config.PROJ_DIM // 2
        )
        self.type_embedding = nn.Embedding(
            num_types,
            Config.PROJ_DIM // 2
        )
        self.days_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        combined_dim = (Config.PROJ_DIM + (Config.PROJ_DIM // 2) + (Config.PROJ_DIM // 2) + 64)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, Config.PROJ_DIM ),
            nn.BatchNorm1d(Config.PROJ_DIM ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.PROJ_DIM , Config.PROJ_DIM  // 2),
            nn.BatchNorm1d(Config.PROJ_DIM  // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.PROJ_DIM  // 2, 1)
        )

    def forward(self,x_text, x_stock, x_type, x_days):
        x_text = self.phobert_proj(x_text)
        x_stock = self.stock_embedding(x_stock)
        x_type = self.type_embedding(x_type)
        x_days = self.days_proj(x_days)
        x_combined = torch.cat((x_text,x_stock,x_type,x_days),dim=1)
        logits = self.classifier(x_combined)
        return logits