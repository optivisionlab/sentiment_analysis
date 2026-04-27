import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from data_preprocessing import process_content_ocr, extract_phobert_embeddings
from dataset import StockNewsDataset
from model import AttentionTimeSeriesSentiment
from train import train_model

def run_pipeline():
    # 1. Load Data
    print("[1] Đọc dữ liệu từ Excel...")
    df = pd.read_excel(config.DATA_PATH, sheet_name=config.SHEET_NAME)
    df = df[['date', 'StockCode', 'Type', 'Title', 'PublishTime', 'Content', 'GPT chấm điểm']]
    
    # 2. Tiền xử lý, OCR và Tokenize
    df = process_content_ocr(df)

    print("Content: ", df['Content_Clean'].iloc[0])
    
    # Sắp xếp cứng theo thời gian để chia Time-Series chuẩn xác
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['StockCode', 'date']).reset_index(drop=True)
    
#     # 3. Trích xuất đặc trưng (PhoBERT)
#     embeddings = extract_phobert_embeddings(df['Full_Text'], device=config.DEVICE, batch_size=config.BATCH_SIZE)
    
#     # 4. Tính toán trọng số lớp
#     neg_count = df['Binary_Target'].value_counts().get(0.0, 0)
#     pos_count = df['Binary_Target'].value_counts().get(1.0, 0)
#     pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32) if pos_count > 0 else torch.tensor(1.0)
    
#     # 5. Cắt tập Train/Test theo cột mốc thời gian (Ví dụ: dữ liệu trước tháng 11 làm Train, sau tháng 11 làm Test)
#     split_date = pd.to_datetime('2025-11-01')
#     train_idx = df[df['date'] < split_date].index
#     test_idx = df[df['date'] >= split_date].index

#     df_train, emb_train = df.loc[train_idx].reset_index(drop=True), embeddings[train_idx]
#     df_test, emb_test = df.loc[test_idx].reset_index(drop=True), embeddings[test_idx]

#     # 6. Tạo DataLoader
#     train_dataset = StockNewsDataset(df_train, emb_train, seq_length=config.SEQ_LENGTH)
#     test_dataset = StockNewsDataset(df_test, emb_test, seq_length=config.SEQ_LENGTH)

#     train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

#     # 7. Khởi tạo & Training
#     model = AttentionTimeSeriesSentiment(
#         input_size=768, hidden_size=config.HIDDEN_SIZE, 
#         num_layers=config.NUM_LAYERS, dropout=config.DROPOUT
#     ).to(config.DEVICE)

#     print("\n[!] Bắt đầu Training Attention-LSTM...")
#     train_model(
#         model, train_loader, test_loader, 
#         epochs=config.EPOCHS, lr=config.LEARNING_RATE, 
#         device=config.DEVICE, pos_weight=pos_weight, weight_decay=config.WEIGHT_DECAY
#     )

if __name__ == "__main__":
    run_pipeline(),