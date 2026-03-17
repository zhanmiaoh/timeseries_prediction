import numpy as np
import matplotlib.pyplot as plt

# def plot_loss(history, model_name, save_path=None):

#     plt.figure(figsize=(8, 5))
#     plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
#     plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
#     plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label=f"Best Epoch: {history['best_epoch']}")
#     plt.title(f"{model_name} Training and Validation Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')

#     plt.show()

def plot_loss_subplots(lstm_history, transformer_history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # LSTM subplot
    axes[0].plot(lstm_history['epoch'], lstm_history['train_loss'], label='Train Loss')
    axes[0].plot(lstm_history['epoch'], lstm_history['val_loss'], label='Val Loss')
    axes[0].axvline(x=lstm_history['best_epoch'], color='r', linestyle='--', label=f"Best Epoch: {lstm_history['best_epoch']}")
    axes[0].set_title("LSTM Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Transformer subplot
    axes[1].plot(transformer_history['epoch'], transformer_history['train_loss'], label='Train Loss')
    axes[1].plot(transformer_history['epoch'], transformer_history['val_loss'], label='Val Loss')
    axes[1].axvline(x=transformer_history['best_epoch'], color='g', linestyle='--', label=f"Best Epoch: {transformer_history['best_epoch']}")
    axes[1].set_title("Transformer Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yticks(axes[0].get_yticks())  # Ensure y-ticks are the same for both subplots
    axes[1].legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

# def plot_loss_2(history_lstm, history_transformer, save_path=None):

#     plt.figure(figsize=(8, 5))
#     plt.plot(history_lstm['epoch'], history_lstm['train_loss'], label='LSTM Train Loss')
#     plt.plot(history_lstm['epoch'], history_lstm['val_loss'], label='LSTM Val Loss')
#     plt.axvline(x=history_lstm['best_epoch'], color='r', linestyle='--', label=f"LSTM Best Epoch: {history_lstm['best_epoch']}")

#     plt.plot(history_transformer['epoch'], history_transformer['train_loss'], label='Transformer Train Loss')
#     plt.plot(history_transformer['epoch'], history_transformer['val_loss'], label='Transformer Val Loss')
#     plt.axvline(x=history_transformer['best_epoch'], color='g', linestyle='--', label=f"Transformer Best Epoch: {history_transformer['best_epoch']}")

#     plt.title(f"Training and Validation Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')

#     plt.show()


def data_prep_plot(timeseries, preds, seq_length):

    total_length = len(timeseries)
    num_preds, horizon = preds.shape

    test_plot = np.full((total_length, horizon), np.nan)

    # num_preds = total_length - seq_length - horizon + 1
    for h in range(horizon):
        start_idx = seq_length + h 
        end_idx = start_idx + num_preds

        test_plot[start_idx:end_idx, h] = preds[:, h]
        
    return test_plot

# on test set: 
# naive_preds, ma_preds, lstm_preds, transformer_preds: (num_samples, 1)
# timeseries: (dates, 1) 


def plot_predictions(timeseries, prediction_dict, config, save_path=None):

    # 使用 .get() 的好处是，万一字典里漏写了某个键，它会返回默认值而不会报错
    seq_length = config.get("seq_length", None)
    ticker = config.get("ticker", "Unknown")
    start_date = config.get("start_date", "Unknown")
    end_date = config.get("end_date", "Unknown")

    target_name = config["target"]

    plt.figure(figsize=(12, 4))
    # true:
    plt.plot(timeseries, label=f'True {target_name}', color='black', linewidth=1.5)
    # prediction:
    for label, preds in prediction_dict.items():
        test_plot = data_prep_plot(timeseries, preds, seq_length)
        plt.plot(test_plot, label=label, alpha=0.8) 

    plt.legend()
    plt.title(f"{ticker} {target_name} from {start_date} to {end_date} (Test set)")
    plt.xlabel("Date")
    plt.ylabel(f"{target_name}")

    if target_name in {"return", "log_return"}:
        plt.axhline(0.0, linestyle="--", alpha=0.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def return_convert_close_plot(test_df_raw, y_test_raw, prediction_dict_raw, config, save_path=None):

    ticker = config.get("ticker", "Unknown")
    seq_length = config.get("seq_length", None)

    plt.figure(figsize=(12, 4))
    # true:
    close_all = test_df_raw["Close"].values.reshape(-1, 1)
    true_close = close_all[seq_length :]          # actual C_{t+1}
    plt.plot(true_close, label="True Close", color="black")

    for model in ["Linear Regression" ,"LSTM", "Transformer"]:
        # pred:
        pred_ret = prediction_dict_raw[model]   # raw predicted return
        true_ret = y_test_raw                    # raw true return

        prev_close = close_all[seq_length - 1 : -1]   # C_t
        pred_close = prev_close * (1 + pred_ret)      # C_{t+1}^pred = C_t * (1+r_{t+1}^pred)
        plt.plot(pred_close, label=f"Pred Close from Pred Return: {model}", alpha=0.8)

    plt.legend()
    plt.title(f"{ticker} Close price converted from predicted returns")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()