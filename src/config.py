import torch

class Config:
    # Paths
    DATA_PATH1 = "/home/chuanh/hoangchu/sentiment_analysis/data/20260123 - Sentiment label.xlsx"
    DATA_PATH2 = "/home/chuanh/hoangchu/sentiment_analysis/data/cafef_vn30_news_h2_2025_sentiment.xlsx"
    DATA_PATH3 = "/home/chuanh/hoangchu/sentiment_analysis/data/fireant_vn30_news_h2_2025_sentiment.xlsx"
    CORENLP_DIR = "/home/chuanh/hoangchu/sentiment_analysis/CoreNLP"
    SHEET_NAME = "Kết quả loại trùng"
    LOG_FILE = "../results"
    MODEL_PATH = "../weights"
    SAVE_PLOTS = "../results/plots"
    TRAIN_RATIO = 0.8
    PHOBERT_MODEL = "vinai/phobert-large"
    PHOBERT_DIM = 1024
    PROJ_DIM = 512
    META_DIM = 64
    BATCH_SIZE = 32
    EPOCHS = 60
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    K_FOLDS = 5
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")