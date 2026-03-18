import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_error_summary(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_pred - y_true

    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mape(%)": float(np.mean(np.abs(residuals / (y_true + 1e-8))) * 100), 
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        # "directional_acc": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
    }


def plot_residual_histogram(y_true, y_pred, bins=50, title="Residual Histogram", save_path=None):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=bins, edgecolor="black")
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.axvline(residuals.mean(), linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Residual = pred - true")
    ax.set_ylabel("Count")

    ax.text(
        0.02, 0.98,
        f"mean = {residuals.mean():.6f}\nstd = {residuals.std():.6f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return residuals


def _topk_non_overlapping_indices(abs_err, k=5, min_gap=5):
    """
    Find the indices of the tpo-k largest errors, 
    while ensuring that selected indices are at least `min_gap` apart.
    min_gap: avoid selecting many adjacent points caused by overlapping windows
    """
    order = np.argsort(abs_err)[::-1] # *indices* (arg) of errors sorted in descending order
    selected = []

    for idx in order:
        if all(abs(idx - s) > min_gap for s in selected):
            selected.append(int(idx))
            if len(selected) == k:
                break
    return selected


def plot_topk_error_intervals(
    y_true,
    y_pred,
    dates=None,
    k=5,
    context=10,
    min_gap=5,
    out_dir=None,
    prefix="topk_error"
):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    abs_err = np.abs(y_pred - y_true)

    topk_idxs = _topk_non_overlapping_indices(abs_err, k=k, min_gap=min_gap)

    if dates is None:
        dates = np.arange(len(y_true))
    else:
        dates = np.asarray(dates)

    rows = []

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for rank, idx in enumerate(topk_idxs, start=1):
        left = max(0, idx - context)
        right = min(len(y_true), idx + context + 1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates[left:right], y_true[left:right], label="True")
        ax.plot(dates[left:right], y_pred[left:right], label="Pred")
        ax.axvline(dates[idx], linestyle="--", linewidth=1)

        ax.set_title(f"Top-{rank} Error Interval | idx={idx} | abs_err={abs_err[idx]:.6f}")
        ax.set_ylabel("Price")
        ax.legend()

        ax2 = ax.twinx()
        returns = np.diff(y_true[left:right], prepend=y_true[left])  # simple return
        ax2.plot(dates[left:right], returns, color='k', alpha=0.2, label="return (volatility)")
        ax2.set_ylabel("Return")
        ax2.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()

        if out_dir is not None:
            save_path = os.path.join(out_dir, f"{prefix}_rank{rank}_idx{idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

        rows.append({
            "rank": rank,
            "idx": int(idx),
            "date": dates[idx],
            "y_true": float(y_true[idx]),
            "y_pred": float(y_pred[idx]),
            "residual": float(y_pred[idx] - y_true[idx]),
            "abs_err": float(abs_err[idx]),
        })

    return pd.DataFrame(rows), topk_idxs


# how to use
# import numpy as np
# import pandas as pd
from plotting import plot_predictions
import json

if __name__ == "__main__":

    analysis_model = "Naive Baseline"
    exp_name = "close_full_seq60_h1"
    run_id = "20260315_180055"
    run_dir = f"outputs/{exp_name}/{run_id}"

    data = np.load(f"{run_dir}/predictions.npz", allow_pickle=True)
    timeseries = data["all_test_timeseries"]
    y_test_raw = data["y_test_raw"]
    prediction_dict_raw = {}
    for key in data.files:
        if key.startswith(("Naive Baseline", "Moving Average", "Linear Regression", "LSTM", "Transformer")):
            prediction_dict_raw[key] = data[key]
    # dates = pd.to_datetime(data["dates"]) if "dates" in data.files else None

    y_true = y_test_raw
    y_pred = prediction_dict_raw[analysis_model]

    summary = compute_error_summary(y_true, y_pred)
    print(summary)

    plot_residual_histogram(
        y_true,
        y_pred,
        title="Close Residual Histogram",
        save_path=f"{run_dir}/figures/{analysis_model}_residual_histogram.png"
    )

    topk_df, topk_idxs = plot_topk_error_intervals(
        y_true,
        y_pred,
        # dates=dates,
        k=5,
        context=10,
        min_gap=5,
        out_dir=f"{run_dir}/figures",
        prefix=f"{analysis_model}_close_price_topk"
    )

    config_path = f"{run_dir}/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    save_path = f"{run_dir}/figures/{config.get('ticker', 'Unknown')}_predictions_with_{analysis_model}_topk.png"
    plot_predictions(timeseries, prediction_dict_raw, config, save_path, topk_idxs)

    print(topk_df)