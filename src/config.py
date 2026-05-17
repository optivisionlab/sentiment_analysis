import torch


class Config:
    # Paths
    DATA_PATH = "data/20260123 - Sentiment label.xlsx"
    SHEET_NAME = "Kết quả loại trùng"
    CORENLP_DIR = "/home/chuanh/quanlm/sentiment_analysis/CoreNLP"
    LOG_FILE = "../results"
    MODEL_PATH = "../weights"
    # Data split
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    SEQ_LEN = 5

    # Text model
    PHOBERT_MODEL = "vinai/phobert-base-v2"
    MAX_LENGTH = 256
    PHOBERT_DIM = 768

    # Model size
    META_DIM = 64
    HIDDEN_SIZE = 128
    MLP_HIDDEN = 256

    # Training
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    NUM_WORKERS = 2

    # Misc
    RANDOM_STATE = 42
    DROP_LAST = False

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")