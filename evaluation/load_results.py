import pandas as pd
import numpy as np
import joblib


# 0) config
# 1) summary metrics
# 2) predictions and y_test
# 3) training history
# 4) checkpoints
# 5) scalers


run_dir = "outputs/20260314_002451"


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


# checkpoints


# scaler
targer_scaler = joblib.load(f"{run_dir}/target_scaler.joblib")
feature_scaler = joblib.load(f"{run_dir}/feature_scaler.joblib")

