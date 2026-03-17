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
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "directional_acc": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
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
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
    return residuals


def _topk_non_overlapping_indices(abs_err, k=5, min_gap=5):
    """
    min_gap: avoid selecting many adjacent points caused by overlapping windows
    """
    order = np.argsort(abs_err)[::-1]
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

    idxs = _topk_non_overlapping_indices(abs_err, k=k, min_gap=min_gap)

    if dates is None:
        dates = np.arange(len(y_true))
    else:
        dates = np.asarray(dates)

    rows = []

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for rank, idx in enumerate(idxs, start=1):
        left = max(0, idx - context)
        right = min(len(y_true), idx + context + 1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates[left:right], y_true[left:right], label="True")
        ax.plot(dates[left:right], y_pred[left:right], label="Pred")
        ax.axvline(dates[idx], linestyle="--", linewidth=1)

        ax.set_title(f"Top-{rank} Error Interval | idx={idx} | abs_err={abs_err[idx]:.6f}")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        if out_dir is not None:
            save_path = os.path.join(out_dir, f"{prefix}_rank{rank}_idx{idx}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

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

    return pd.DataFrame(rows)


# how to use
# import numpy as np
# import pandas as pd

# data = np.load("outputs/your_run_id/predictions.npz", allow_pickle=True)

# y_true = data["y_true"]
# y_pred = data["y_pred"]
# dates = pd.to_datetime(data["dates"]) if "dates" in data.files else None

# summary = compute_error_summary(y_true, y_pred)
# print(summary)

# plot_residual_histogram(
#     y_true,
#     y_pred,
#     title="Return Residual Histogram",
#     out_path="outputs/your_run_id/figures/residual_histogram.png"
# )

# topk_df = plot_topk_error_intervals(
#     y_true,
#     y_pred,
#     dates=dates,
#     k=5,
#     context=10,
#     min_gap=5,
#     out_dir="outputs/your_run_id/figures",
#     prefix="return_topk"
# )

# print(topk_df)