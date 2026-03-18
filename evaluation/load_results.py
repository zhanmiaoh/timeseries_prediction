import pandas as pd
import numpy as np
import json
import joblib


# 0) config
# 1) summary metrics
# 2) predictions and y_test
# 3) training history
# 4) checkpoints
# 5) scalers

exp_name = "close_full_seq60_h1"
run_id = "20260315_180055"
run_dir = f"outputs/{exp_name}/{run_id}"

# config
config_path = f"{run_dir}/config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"Load config, ticker: {config.get('ticker')}")


# summary_df
summary_df = pd.read_csv(f"{run_dir}/summary_metrics.csv")
print(summary_df.head())



# predictions and y_test
data = np.load(f"{run_dir}/predictions.npz")
timeseries = data["all_test_timeseries"]
y_test_raw = data["y_test_raw"]
prediction_dict_raw = {}
for key in data.files:
    if key.startswith(("Naive Baseline", "Moving Average", "Linear Regression", "LSTM", "Transformer")):
        prediction_dict_raw[key] = data[key]


# training history (read csv back to dict: history_lstm, history_transformer)
df = pd.read_csv(f"{run_dir}/history_lstm.csv")
history_lstm = {
    "epoch": df["epoch"].tolist(),
    "train_loss": df["train_loss"].tolist(),
    "val_loss": df["val_loss"].tolist(),
    "best_epoch": int(df["best_epoch"].iloc[0]),
    "best_val_loss": float(df["best_val_loss"].iloc[0])
}

df = pd.read_csv(f"{run_dir}/history_transformer.csv")
history_transformer = {
    "epoch": df["epoch"].tolist(),
    "train_loss": df["train_loss"].tolist(),
    "val_loss": df["val_loss"].tolist(),
    "best_epoch": int(df["best_epoch"].iloc[0]),
    "best_val_loss": float(df["best_val_loss"].iloc[0])
}


# scaler
targer_scaler = joblib.load(f"{run_dir}/target_scaler.joblib")
feature_scaler = joblib.load(f"{run_dir}/feature_scaler.joblib")



# checkpoints
import torch
from models.lstm import LSTMModel
from models.transformer import TransformerForecast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load LSTM
# load checkpoint into device (cpu/cuda)
ckpt_path = f"{run_dir}/lstm_best.pt"
lstm_ckpt = torch.load(ckpt_path, map_location=torch.device(device))

# create object
saved_config = lstm_ckpt.get("config", {}) 
lstm_params = saved_config.get("lstm_params", {})
lstm_model = LSTMModel(**lstm_params)

# load state dict into model
lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])


# load transformer
ckpt_path = f"{run_dir}/transformer_best.pt"
transformer_ckpt = torch.load(ckpt_path, map_location=torch.device(device))

saved_config = transformer_ckpt.get("config", {})
transformer_params = saved_config.get("transformer_params", {})
transformer_model = TransformerForecast(**transformer_params)

transformer_model.load_state_dict(transformer_ckpt["model_state_dict"])
