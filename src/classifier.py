import torch
import torch.nn as nn
from src.config import Config


class SentimentModel(nn.Module):
    
    def __init__(self, num_stocks, num_types):
        super().__init__()
        self.stock_emb = nn.Embedding(num_stocks, Config.META_DIM)
        self.type_emb = nn.Embedding(num_types, Config.META_DIM)
        lstm_input_dim = Config.PHOBERT_DIM + (Config.META_DIM * 2)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(Config.HIDDEN_SIZE * 2, Config.MLP_HIDDEN),
            nn.LayerNorm(Config.MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.MLP_HIDDEN, 1)
        )

    def forward(self, x_text_emb, x_stock, x_type):
        s_emb = self.stock_emb(x_stock)     
        t_emb = self.type_emb(x_type)

        combined_sequence = torch.cat((x_text_emb, s_emb, t_emb), dim=2)
        _, (h_n, _) = self.lstm(combined_sequence)

        last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        logits = self.classifier(last_hidden)
        return logits