import numpy as np
from src.config import Config
from src.dataset import (
    load_and_preprocess_data,
    extract_phobert_embeddings,
    StockSequenceDataset
)
from src.train import fit_model, test_model
from src.utils import save_logs, plot_metrics


def main():
    full_df, train_df, val_df, test_df, num_stocks, num_types, stock2id, type2id = load_and_preprocess_data()
    print("Extracting text embeddings for toàn bộ dataset...")
    text_embeddings = extract_phobert_embeddings(full_df["Title"].tolist())

    train_dataset = StockSequenceDataset(train_df, text_embeddings, seq_len=Config.SEQ_LEN)
    val_dataset = StockSequenceDataset(val_df, text_embeddings, seq_len=Config.SEQ_LEN)
    test_dataset = StockSequenceDataset(test_df, text_embeddings, seq_len=Config.SEQ_LEN)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    model, logs = fit_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_stocks=num_stocks,
        num_types=num_types
    )

    df_logs = save_logs(logs)
    plot_metrics(df_logs)

    result = test_model(model, test_dataset)
    print("\nTest result:")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()