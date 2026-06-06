from src.dataloader import load_and_preprocess_data, extract_phobert_embeddings
from src.train import train_kfold, test
from src.utils import save_logs, plot_metrics
from src.config import Config
import os
import py_vncorenlp


def main():

    print("Preprocessing Data ...")
    train_df, test_df, _, _, _, num_stocks, num_types = load_and_preprocess_data()
    print(train_df)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    X_text_train = extract_phobert_embeddings(train_df['Title'].tolist())
    X_stock_train = train_df['StockID'].values
    X_type_train = train_df['TypeID'].values
    X_days_train = train_df['days_from_start'].values 
    Y_train = train_df['Target'].values
    # download model nếu chưa tồn tại
    print("\nTrain KFold ...")
    best_model, logs, fold_summary_df = train_kfold(
        X_text_train, X_stock_train, X_type_train, X_days_train, 
        Y_train, num_stocks, num_types
    )
    
    print("\nSaving Result ...")
    df_logs = save_logs(logs, name = "training_logs.xlsx")
    plot_metrics(df_logs, Config.SAVE_PLOTS)
    df_logs2 = save_logs(fold_summary_df, "Summary.xlsx")
    print("\nTest Finall...")
    X_text_test = extract_phobert_embeddings(test_df['Title'].tolist())
    result_test = test(
        model=best_model,
        X_text=X_text_test,
        X_stock=test_df['StockID'].values,
        X_type=test_df['TypeID'].values,
        X_days=test_df['days_from_start'].values,
        Y_target=test_df['Target'].values,
    )
    print(result_test)

if __name__ == "__main__":
    main()

