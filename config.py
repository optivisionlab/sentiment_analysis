import torch

# Paths
DATA_PATH = "20260123 - Sentiment label.xlsx"
SHEET_NAME = "Kết quả loại trùng"
BASE_DIR = "/content/"

# Hyperparameters
MAX_LEN = 256
BATCH_SIZE = 32
SEQ_LENGTH = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')