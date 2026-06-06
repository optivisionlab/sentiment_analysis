import pandas as pd
import numpy as np
from typing import Dict, Tuple
import torch
import py_vncorenlp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from src.config import Config
import os


if not os.path.exists(Config.CORENLP_DIR):
        py_vncorenlp.download_model(save_dir=Config.CORENLP_DIR)

rdrsegmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir=Config.CORENLP_DIR
)

class StockDataset(Dataset):
    def __init__(self, X_text, X_stock, X_type, X_days, y):
        self.X_text = torch.tensor(X_text, dtype=torch.float32)
        self.X_stock = torch.tensor(X_stock, dtype=torch.long)
        self.X_type = torch.tensor(X_type, dtype=torch.long)
        self.X_days = torch.tensor(X_days, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_text[idx], self.X_stock[idx], self.X_type[idx], self.X_days[idx], self.y[idx]
    
def split_per_stock(data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_list = []
        val_list = []
        for s, g in data.groupby('StockID', sort=False):
            n = len(g)
            cut = int(n * train_ratio)
            train_list.append(g.iloc[:cut].copy())
            val_list.append(g.iloc[cut:].copy())
        train_df = pd.concat(train_list, axis=0).sort_values(['StockID', 'date']).reset_index(drop=True)
        val_df = pd.concat(val_list, axis=0).sort_values(['StockID', 'date']).reset_index(drop=True)
        return train_df, val_df
    
def load_and_preprocess_data():
    # Load data
    df1 = pd.read_excel(Config.DATA_PATH1, sheet_name=Config.SHEET_NAME)
    df2 = pd.read_excel(Config.DATA_PATH2)
    df3 = pd.read_excel(Config.DATA_PATH3)

    rename_cols = {
        "Mã CK": "StockCode",
        "Ngày": "date",
        "Tiêu đề": "Title",
        "Điểm Sentiment (0-10)": "GPT chấm điểm"
    }
    df2 = df2.rename(columns=rename_cols)
    df3 = df3.rename(columns=rename_cols)
    df2["Type"] = "Tin mới nhất"
    df3["Type"] = "Tin mới nhất"
    df1["date"] = pd.to_datetime(
        df1["date"],
        format="%m/%d/%y",
        errors="coerce"
    )
    df2["date"] = pd.to_datetime(
        df2["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df3["date"] = pd.to_datetime(
        df3["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )
    columns = [
        "date",
        "StockCode",
        "Type",
        "Title",
        "GPT chấm điểm"
    ]
    df1 = df1[columns]
    df2 = df2[columns]
    df3 = df3[columns]

    df = pd.concat(
        [df1, df2, df3],
        ignore_index=True
    )
    df["Title"] = (
        df["Title"]
        .astype(str)
        .str.strip()
    )
    df = df.drop_duplicates(
        subset=["Title"],
        keep="first"
    )
    df = df.reset_index(drop=True)
    df = df[['date', 'StockCode', 'Type', 'Title', 'GPT chấm điểm']]
    df["Target"] = (df["GPT chấm điểm"] > 5).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    
    df = df.sort_values(['StockCode', 'date']).reset_index(drop=True)
    
    # Map Stock to ID & Type to ID (Label Encoding)
    stocks = df['StockCode'].unique().tolist()
    stock2id: Dict[str, int] = {s: i for i, s in enumerate(stocks)}
    id2stock: Dict[int, str] = {i: s for s, i in stock2id.items()}
    df['StockID'] = df['StockCode'].map(stock2id)
    types = df['Type'].unique().tolist()
    type2id: Dict[str, int] = {t: i for i, t in enumerate(types)}
    df['TypeID'] = df['Type'].map(type2id)
    
    # Add feature "days_from_start"
    min_date = df["date"].min()
    df["days_from_start"] = (df["date"] - min_date).dt.days
    
    # Train_Test_Split
    train_df, test_df = split_per_stock(df, Config.TRAIN_RATIO)    
    
    # Scale Data
    scalers: Dict[int, MinMaxScaler] = {}
    train_scaled_parts = []
    for s, g in train_df.groupby('StockID', sort=False):
        scaler = MinMaxScaler()
        g_scaled = g.copy()
        g_scaled["days_from_start"] = scaler.fit_transform(g[["days_from_start"]])
        train_scaled_parts.append(g_scaled)
        scalers[s] = scaler
    train_df = pd.concat(train_scaled_parts, axis=0).sort_values(['StockID', 'date']).reset_index(drop=True)
    test_scaled_parts = []
    for s, g in test_df.groupby('StockID', sort=False):
        g_scaled = g.copy()
        if s in scalers:
            g_scaled["days_from_start"] = scalers[s].transform(g[["days_from_start"]])
        else:
            g_scaled["days_from_start"] = 0 
        test_scaled_parts.append(g_scaled)
    test_df = pd.concat(test_scaled_parts, axis=0).sort_values(['StockID', 'date']).reset_index(drop=True)
    num_stocks = len(stock2id)
    num_types = len(type2id)

    return train_df, test_df, stock2id, id2stock, type2id, num_stocks, num_types


def extract_phobert_embeddings(texts):
    segmented_texts = [rdrsegmenter.word_segment(t)[0] for t in texts]
    #embedding word
    print("Loading PhoBERT")
    tokenizer = AutoTokenizer.from_pretrained(Config.PHOBERT_MODEL)
    phobert = AutoModel.from_pretrained(Config.PHOBERT_MODEL).to(Config.DEVICE)
    phobert.eval()
    
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(segmented_texts), Config.BATCH_SIZE), desc="Extracting Embeddings"):
            batch_text = segmented_texts[i:i + Config.BATCH_SIZE]
            text_token = tokenizer(batch_text, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(Config.DEVICE) for k, v in text_token.items()}
            model_output = phobert(**encoded_input)
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled_embeddings = sum_embeddings / sum_mask
            embeddings.append(mean_pooled_embeddings.cpu().numpy())
    return np.vstack(embeddings)