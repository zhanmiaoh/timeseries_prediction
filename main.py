import torch
import numpy as np
import matplotlib.pyplot as plt

from data.data_loader import load_stock_data
from data.preprocess import engineer_features, create_split, scale_features, target_inverse_scale
from data.dataset import StockDataset, StockDataloader, loader_to_numpy

from models.baselines import naive_forecast, moving_average_forecast, linear_regression_forecast
from models.lstm import LSTMModel
from models.transformer import TransformerForecast
from training.train import train_model, predict_model

from evaluation.metrics import mae, rmse, mape
from evaluation.plotting import data_prep_plot, plot_predictions, plot_loss_subplots
from evaluation.save_results import save_results

from datetime import datetime
import sys
import os


def main():

    # log_file = open("outputs/log.txt", "a")
    # sys.stdout = log_file

    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2025-01-01"
    TRAIN_END = "2020-12-31"
    VAL_END = "2022-12-31"
    SEQ_LENGTH = 60
    HORIZON = 1
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-3

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}]" ,f"\nUsing device: {device}\n")

    # Set seed
    seed = 27
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None


    # load data
    df = load_stock_data(TICKER, START_DATE, END_DATE)
    print("Raw data shape:", df.shape)

    # define features and target
    # original_cols = df.columns.tolist() 
    original_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    new_features = ["return", "log_return"]
    features = original_features + new_features
    target = "Close"
    target_index = features.index(target)
    print("\nFeatures:", features)
    print("Target:", target, f"Target index in features: {target_index}")
    # print(f"target index in features: {target_index}")

    # engineer features
    df = engineer_features(df, new_features)

    print("\nAfter engineering features:")
    print(df.shape)
    print(df.head())
    print("index type:", type(df.index), "index dtype:", df.index.dtype if hasattr(df.index, "dtype") else "N/A")
    print("index min/max:", df.index.min(), df.index.max())

    # split train/val/test, and scale
    train_df_raw, val_df_raw, test_df_raw = create_split(df, TRAIN_END, VAL_END)
    scaler, target_scaler, train_df, val_df, test_df = scale_features(
        train_df_raw, val_df_raw, test_df_raw, features, target
    )
    # below all scaled data used
    print("\nFinal train/val/test shapes:")
    print("Train data:", train_df.shape, train_df.index.min(), train_df.index.max())
    print("Val data:", val_df.shape, val_df.index.min(), val_df.index.max())
    print("Test data:", test_df.shape, test_df.index.min(), test_df.index.max())


    # create sliding-window dataset and dataloader
    print("\n---Creating datasets...---")
    train_dataset = StockDataset(train_df, features, target, SEQ_LENGTH, HORIZON)
    val_dataset = StockDataset(val_df, features, target, SEQ_LENGTH, HORIZON)
    test_dataset = StockDataset(test_df, features, target, SEQ_LENGTH, HORIZON)
    print(f"Number of sliding-window samples: "
        f" Train: {len(train_dataset)}, Test: {len(test_dataset)}, Val: {len(val_dataset)}")

    train_loader, val_loader, test_loader = StockDataloader(train_dataset, val_dataset, test_dataset, BATCH_SIZE)

    # create numpy data for baseline testing
    X_train, y_train = loader_to_numpy(train_loader)
    X_test, y_test = loader_to_numpy(test_loader)


    # evaluate baseline models
    naive_preds = naive_forecast(X_test, target_index)
    ma_preds = moving_average_forecast(X_test, target_index, window=5)
    lr_preds = linear_regression_forecast(X_train, y_train, X_test)

    print("\n---Baseline Model Evaluation on Test Set (scaled)---")
    print(f"Naive Forecast - MAE: {mae(y_test, naive_preds):.4f}, RMSE: {rmse(y_test, naive_preds):.4f}")
    print(f"Moving Average - MAE: {mae(y_test, ma_preds):.4f}, RMSE: {rmse(y_test, ma_preds):.4f}")
    print(f"Linear Regression - MAE: {mae(y_test, lr_preds):.4f}, RMSE: {rmse(y_test, lr_preds):.4f}")


    print("\n---Training LSTM model...---")
    # create model
    lstm_params = {
        "num_features": len(features), 
        "hidden_size": 64, 
        "num_layers": 2, 
        "dropout": 0.2, 
        "horizon": HORIZON
    }
    lstm_model = LSTMModel(**lstm_params)
    # lstm_model = LSTMModel(
    #     num_features=len(features), 
    #     hidden_size=64, 
    #     num_layers=2, 
    #     dropout=0.2, 
    #     horizon=HORIZON
    # )

    # create optimizer and loss function
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LR)
    lstm_criterion = torch.nn.MSELoss()

    # train and evaluate LSTM
    lstm_history, lstm_model = train_model(
        lstm_model, train_loader, val_loader, lstm_optimizer, lstm_criterion, device, epochs=EPOCHS
    )
    lstm_preds = predict_model(lstm_model, test_loader, device)
    print(f"LSTM Model - (scaled) MAE: {mae(y_test, lstm_preds):.4f}, RMSE: {rmse(y_test, lstm_preds):.4f}")


    print("\n---Training Transformer model...---")
    # create model
    transformer_params = {
        "num_features": len(features), 
        "d_model": 64, 
        "nhead": 4, 
        "num_layers": 2, 
        "dropout": 0.1, 
        "horizon": HORIZON
    }
    transformer_model = TransformerForecast(**transformer_params)
    # transformer_model = TransformerForecast(
    #     num_features=len(features), 
    #     d_model=64, 
    #     nhead=4, 
    #     num_layers=2, 
    #     dropout=0.1, 
    #     horizon=HORIZON
    # )

    # create optimizer and criterion
    transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=LR)
    transformer_criterion = torch.nn.MSELoss()

    # train and evaluate Transformer
    transformer_history, transformer_model = train_model(
        transformer_model, train_loader, val_loader, transformer_optimizer, transformer_criterion, device, epochs=EPOCHS
    )
    transformer_preds = predict_model(transformer_model, test_loader, device)
    print(f"Transformer Model - (scaled) MAE: {mae(y_test, transformer_preds):.4f}, RMSE: {rmse(y_test, transformer_preds):.4f}")

    # getting all preds 
    prediction_dict = {
        'Naive Baseline': naive_preds,
        'Moving Average': ma_preds,
        'Linear Regression': lr_preds,
        'LSTM': lstm_preds,
        'Transformer': transformer_preds
    }
    y_test_raw, prediction_dict_raw = target_inverse_scale(y_test, prediction_dict, target_scaler)



    
    config = {
        "ticker": TICKER,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "seq_length": SEQ_LENGTH,
        "horizon": HORIZON,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "features": features,
        "target": target,
        "seed": seed,
            # model params
        "lstm_params": lstm_params,
        "transformer_params": transformer_params,
    }


    # save and plot
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)

    # plot, and save figure
    timeseries = test_df_raw[target].values.reshape(-1,1) # (num_dates, horizon=1)
    print(f"Total test dates: {timeseries.shape}, Seq_length={SEQ_LENGTH}, Horizon={HORIZON}, Pred dates: {ma_preds.shape}")
    plot_predictions(timeseries, prediction_dict_raw, config, 
                     save_path=f"{run_dir}/figures/{TICKER}_predictions.png")
    
    plot_loss_subplots(lstm_history, transformer_history, 
                       save_path=f"{run_dir}/figures/{TICKER}_lstm_transformer_loss.png")
    
    
    # save outputs: metrics, history, predictions, checkpoints, scalers
    summary_df = save_results(run_dir, config, timeseries, y_test_raw, prediction_dict_raw, 
                              lstm_model, lstm_history, lstm_optimizer, 
                              transformer_model, transformer_history, transformer_optimizer, 
                              scaler, target_scaler)

    
    # print("\n---Final Results Summary (on raw data scale)---")
    # print("=" * 65)
    # print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'MAPE':<15}")
    # print("-" * 65)
    # for label, preds in prediction_dict_raw.items():
    #     print(f"{label:<20} {mae(y_test_raw, preds):<15.4f} {rmse(y_test_raw, preds):<15.4f} {f'{mape(y_test_raw, preds):.4f}%':<15}")
    # print("=" * 65)
    
    print("\n---Final Results Summary (on raw data scale)---")
    print("=" * 65)
    print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'MAPE':<15}")
    print("-" * 65)
    for index, row in summary_df.iterrows():
        mape = row['mape_raw']
        print(f"{row['model']:<20} {row['mae_raw']:<15.4f} {row['rmse_raw']:<15.4f} {f'{mape:.4f}%':<15}")
    print("=" * 65)

    # log_file.close()



if __name__ == "__main__":
    main()