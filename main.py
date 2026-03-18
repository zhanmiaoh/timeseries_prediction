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
from evaluation.plotting import data_prep_plot, plot_predictions, plot_loss_subplots, return_convert_close_plot
from evaluation.save_results import save_results

from datetime import datetime
import sys
import os
import argparse, json


def main():

    # log_file = open("outputs/log.txt", "a")
    # sys.stdout = log_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/close_full_seq60_h1.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    EXP_NAME   = cfg["exp_name"]

    TICKER     = cfg["ticker"]
    START_DATE = cfg["start_date"]
    END_DATE   = cfg["end_date"]
    TRAIN_END  = cfg["train_end"]
    VAL_END    = cfg["val_end"]
    SEQ_LENGTH = cfg["seq_length"]
    HORIZON    = cfg["horizon"]
    BATCH_SIZE = cfg["batch_size"]
    EPOCHS     = cfg["epochs"]
    LR         = cfg["lr"]
    
    # original_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    # new_features = ["return", "log_return"]
    # features = original_features + new_features
    adjust   = cfg["adjust"]
    features   = cfg["features"]
    target     = cfg["target"]
    seed       = cfg["seed"]

    lstm_params = cfg["lstm_params"]
    transformer_params = cfg["transformer_params"]


    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M')}]" ,f"\nUsing device: {device}\n")

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None


    # load data
    df = load_stock_data(TICKER, START_DATE, END_DATE, adjust=adjust)
    print("Raw data shape:", df.shape)

    target_index = features.index(target)
    print("\nFeatures:", features)
    print("Target:", target, f", Target index in features: {target_index}")
    # print(f"target index in features: {target_index}")

    # engineer features
    df = engineer_features(df, new_features=["return", "log_return"])

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
    naive_preds = naive_forecast(X_test, target, target_index)
    ma_preds = moving_average_forecast(X_test, target_index, window=5)
    lr_preds = linear_regression_forecast(X_train, y_train, X_test)

    print("\n---Baseline Model Evaluation on Test Set (scaled)---")
    print(f"Naive Forecast - MAE: {mae(y_test, naive_preds):.4f}, RMSE: {rmse(y_test, naive_preds):.4f}")
    print(f"Moving Average - MAE: {mae(y_test, ma_preds):.4f}, RMSE: {rmse(y_test, ma_preds):.4f}")
    print(f"Linear Regression - MAE: {mae(y_test, lr_preds):.4f}, RMSE: {rmse(y_test, lr_preds):.4f}")


    print("\n---Training LSTM model...---")
    # create model, model params can be modified in config file without changing code here
    # lstm_params = {
    #     "num_features": len(features), 
    #     "hidden_size": 64, 
    #     "num_layers": 2, 
    #     "dropout": 0.2, 
    #     "horizon": HORIZON
    # }
    lstm_model = LSTMModel(**lstm_params)

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
    # transformer_params = {
    #     "num_features": len(features), 
    #     "d_model": 64, 
    #     "nhead": 4, 
    #     "num_layers": 2, 
    #     "dropout": 0.1, 
    #     "horizon": HORIZON
    # }
    transformer_model = TransformerForecast(**transformer_params)

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



    
    # config = {
    #     "ticker": TICKER,
    #     "start_date": START_DATE,
    #     "end_date": END_DATE,
    #     "train_end": TRAIN_END,
    #     "val_end": VAL_END,
    #     "seq_length": SEQ_LENGTH,
    #     "horizon": HORIZON,
    #     "batch_size": BATCH_SIZE,
    #     "lr": LR,
    #     "features": features,
    #     "target": target,
    #     "seed": seed,
    #         # model params
    #     "lstm_params": lstm_params,
    #     "transformer_params": transformer_params,
    # }


    # SAVE and PLOT 
    run_dir = os.path.join("outputs", EXP_NAME, datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)

    # plot loss
    plot_loss_subplots(lstm_history, transformer_history, 
                       save_path=f"{run_dir}/figures/{TICKER}_lstm_transformer_loss.png")

    # plot the target predictions
    timeseries = test_df_raw[target].values.reshape(-1,1) # (num_dates, horizon=1)
    print(f"Total test dates: {timeseries.shape}, Seq_length={SEQ_LENGTH}, Horizon={HORIZON}, Pred dates: {ma_preds.shape}")
    plot_predictions(timeseries, prediction_dict_raw, cfg, 
                     save_path=f"{run_dir}/figures/{TICKER}_predictions.png")
    
    # convert pred return to close, and plot pred close and true close
    if target == "return":
        return_convert_close_plot(test_df_raw, y_test_raw, prediction_dict_raw, cfg, 
                                  save_path=f"{run_dir}/figures/{TICKER}_pred_close_price.png")    
    
    # save outputs: metrics, history, predictions, checkpoints, scalers
    summary_df = save_results(run_dir, cfg, timeseries, y_test_raw, prediction_dict_raw, 
                              lstm_model, lstm_history, lstm_optimizer, 
                              transformer_model, transformer_history, transformer_optimizer, 
                              scaler, target_scaler)

    
    print("\n---Final Results Summary (on raw data scale)---")
    print("=" * 65)
    for index, row in summary_df.iterrows():
        if target in {"return", "log_return"}:
            print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'Direction acc':<15}")
            print("-" * 65)
            print(f"{row['model']:<20} {row['mae_raw']:<15.4f} {row['rmse_raw']:<15.4f} {row['directional_acc']:<15.4f}")
        elif target in {"Close", "Open", "High", "Low"}:
            print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'MAPE':<15}")
            print("-" * 65)
            mape_value = row['mape_raw']
            print(f"{row['model']:<20} {row['mae_raw']:<15.4f} {row['rmse_raw']:<15.4f} {f'{mape_value:.4f}%':<15}")
    print("=" * 65)

    # log_file.close()



if __name__ == "__main__":
    main()