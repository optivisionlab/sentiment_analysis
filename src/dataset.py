import os
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
import py_vncorenlp
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from src.config import Config


class StockSequenceDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, text_embeddings: np.ndarray, seq_len: int = None):
        self.seq_len = seq_len or Config.SEQ_LEN
        X_text, X_stock, X_type, y = [], [], [], []
        for stock_id, g in df.groupby("StockID", sort=False):
            g = g.sort_values(["date", "row_id"]).reset_index(drop=True)
            if len(g) < self.seq_len:
                continue
            row_ids = g["row_id"].to_numpy()
            stock_ids = g["StockID"].to_numpy()
            type_ids = g["TypeID"].to_numpy()
            targets = g["Target"].to_numpy()
            for end in range(self.seq_len - 1, len(g)):
                start = end - self.seq_len + 1
                window_row_ids = row_ids[start:end + 1]
                X_text.append(text_embeddings[window_row_ids])          
                X_stock.append(stock_ids[start:end + 1])                
                X_type.append(type_ids[start:end + 1])                  
                y.append(targets[end])
        self.X_text = torch.tensor(np.array(X_text), dtype=torch.float32)
        self.X_stock = torch.tensor(np.array(X_stock), dtype=torch.long)
        self.X_type = torch.tensor(np.array(X_type), dtype=torch.long)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_text[idx],
            self.X_stock[idx],
            self.X_type[idx],
            self.y[idx]
        )

def split_per_stock_timeseries(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_list, val_list, test_list = [], [], []
    data = data.sort_values(["StockID", "date", "row_id"]).reset_index(drop=True)
    for stock_id, g in data.groupby("StockID", sort=False):
        g = g.sort_values(["date", "row_id"]).reset_index(drop=True)
        n = len(g)
        if n < 3:
            continue
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_list.append(g.iloc[:train_end].copy())
        val_list.append(g.iloc[train_end:val_end].copy())
        test_list.append(g.iloc[val_end:].copy())

    train_df = pd.concat(train_list, axis=0).sort_values(["StockID", "date", "row_id"]).reset_index(drop=True)
    val_df = pd.concat(val_list, axis=0).sort_values(["StockID", "date", "row_id"]).reset_index(drop=True)
    test_df = pd.concat(test_list, axis=0).sort_values(["StockID", "date", "row_id"]).reset_index(drop=True)
    return train_df, val_df, test_df

def load_and_preprocess_data():
    df = pd.read_excel(Config.DATA_PATH, sheet_name=Config.SHEET_NAME)
    df = df[['date', 'StockCode', 'Type', 'Title', 'ArticleID', 'GPT chấm điểm']].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["Target"] = (df["GPT chấm điểm"] > 5).astype(int)

    df = df.sort_values(["StockCode", "date", "ArticleID"]).reset_index(drop=True)

    # row_id dùng để map embedding sau này
    df["row_id"] = np.arange(len(df))

    # Label encoding
    stocks = df["StockCode"].unique().tolist()
    stock2id: Dict[str, int] = {s: i for i, s in enumerate(stocks)}
    type_list = df["Type"].unique().tolist()
    type2id: Dict[str, int] = {t: i for i, t in enumerate(type_list)}
    df["StockID"] = df["StockCode"].map(stock2id)
    df["TypeID"] = df["Type"].map(type2id)

    train_df, val_df, test_df = split_per_stock_timeseries(
        df,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO
    )
    num_stocks = len(stock2id)
    num_types = len(type2id)
    return df, train_df, val_df, test_df, num_stocks, num_types, stock2id, type2id


def extract_phobert_embeddings(texts):
    os.makedirs(Config.CORENLP_DIR, exist_ok=True)
    jar_path = os.path.join(Config.CORENLP_DIR, "VnCoreNLP-1.2.jar")
    if not os.path.exists(jar_path):
        py_vncorenlp.download_model(save_dir=Config.CORENLP_DIR)
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=Config.CORENLP_DIR)
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