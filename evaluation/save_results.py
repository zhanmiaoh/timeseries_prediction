import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import torch

from evaluation.metrics import mae, rmse, mape


def save_results(run_dir, config, timeseries, y_test_raw, prediction_dict_raw, lstm_model, lstm_history, lstm_optimizer, 
                 transformer_model, transformer_history, transformer_optimizer, scaler, target_scaler,
                 dates=None):
    """
    # 0) config.json
    # 1) summary_metrics.csv
    # 2) predictions and y_test: predictions.npz 
    # 3) training history: history_lstm.csv, history_transformer.csv
    # 4) checkpoints: lstm_best.pt, transformer_best.pt
    # 5) scalers: target_scaler.joblib, feature_scaler.joblib
    """

    run_id = os.path.basename(run_dir)


    # 0) save config
    with open(f"{run_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


    # 1) summary metrics
    # y_test_raw, prediction_dict_raw 的summary metrics
    summary_rows = []
    for label, preds in prediction_dict_raw.items():
        sign_acc = (np.sign(y_test_raw) == np.sign(preds)).mean() * 100
        summary_rows.append({
            "run_id": run_id,
            "ticker": config.get("ticker", "Unknown"),
            "start_date": config.get("start_date", "Unknown"),
            "train_end": config.get("train_end", "Unknown"),
            "val_end": config.get("val_end", "Unknown"),
            "seq_length": config.get("seq_length", None),
            "horizon": config.get("horizon", None),
            "model": label,
            "mae_raw": mae(y_test_raw, preds),
            "rmse_raw": rmse(y_test_raw, preds),
            "mape_raw": mape(y_test_raw, preds),
            "directional_acc": sign_acc
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(run_dir, "summary_metrics.csv"), index=False)


    # 2) save predictions (raw): y_test_raw, prediction_dict_raw; can be used for plotting
    save_dict = {"all_test_timeseries": timeseries, "y_test_raw": y_test_raw, **prediction_dict_raw}
    if dates is not None:
        save_dict["dates"] = np.array(dates)
    np.savez(f"{run_dir}/predictions.npz", **save_dict) # save with each variable in save_dict


    # 3) training history
    # 把lstm和transformer串到同一个？
    lstm_history_df = pd.DataFrame({
        "epoch": lstm_history["epoch"],
        "train_loss": lstm_history["train_loss"],
        "val_loss": lstm_history["val_loss"],
    })
    lstm_history_df["model"] = "LSTM"
    lstm_history_df["best_epoch"] = lstm_history["best_epoch"]
    lstm_history_df["best_val_loss"] = lstm_history["best_val_loss"]
    lstm_history_df.to_csv(f"{run_dir}/history_lstm.csv", index=False)


    transformer_history_df = pd.DataFrame({
        "epoch": transformer_history["epoch"], # multi-rows, epochs 1,2,... in col "epoch"
        "train_loss": transformer_history["train_loss"], 
        "val_loss": transformer_history["val_loss"],
    })
    transformer_history_df["model"] = "Transformer"  # will be applied to all rows
    transformer_history_df["best_epoch"] = transformer_history["best_epoch"]
    transformer_history_df["best_val_loss"] = transformer_history["best_val_loss"]
    transformer_history_df.to_csv(f"{run_dir}/history_transformer.csv", index=False)


    # 4) checkpoints
    torch.save({
        "model_state_dict": lstm_model.state_dict(),
        "optimizer_state_dict": lstm_optimizer.state_dict(),
        "history": lstm_history,
        "config": config,
    }, f"{run_dir}/lstm_best.pt")

    torch.save({
        "model_state_dict": transformer_model.state_dict(),
        "optimizer_state_dict": transformer_optimizer.state_dict(),
        "history": transformer_history,
        "config": config,
    }, f"{run_dir}/transformer_best.pt")


    # 5) scalers
    joblib.dump(target_scaler, f"{run_dir}/target_scaler.joblib")
    joblib.dump(scaler, f"{run_dir}/feature_scaler.joblib")
    

    return summary_df
    

