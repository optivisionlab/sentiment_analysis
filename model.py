import torch
import torch.nn as nn

class AttentionTimeSeriesSentiment(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=2, dropout=0.3, num_classes=1):
        super(AttentionTimeSeriesSentiment, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, 
            num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Mạng Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # lstm_out: (batch, seq, hidden)
        lstm_out, _ = self.lstm(x)
        
        # Tính attention scores
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Tính context vector
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out