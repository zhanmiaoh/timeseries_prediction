
import matplotlib.pyplot as plt
import numpy as np

# ========== edit these ==========
FIG_W = 6.2   # inches
FIG_H = 3.4   # inches
DPI = 240

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 11
# ================================

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

# ---------------------------
# 1) Feature ablation
# ---------------------------
feature_configs = ["Full", "Only return", "Only log", "OHLCV"]
lstm_mae = np.array([2.6038, 2.1001, 2.1048, 3.8329])
transformer_mae = np.array([2.4518, 4.3076, 2.7707, 3.1877])
baseline_feature = 1.9045

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
x = np.arange(len(feature_configs))
width = 0.32

ax.bar(x - width/2, lstm_mae, width=width, label="LSTM")
ax.bar(x + width/2, transformer_mae, width=width, label="Transformer")
ax.axhline(baseline_feature, linestyle="--", linewidth=1.6, label="Naive")
ax.set_xticks(x)
ax.set_xticklabels(feature_configs)
ax.set_ylabel("MAE")
# ax.set_title("Feature Ablation")
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10))
ax.set_ylim(0, 4.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("feature_ablation_compact.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------
# 2) Sequence length ablation
# ---------------------------
seq_lengths = np.array([20, 60, 120])
linear_seq_mae = np.array([3.8090, 5.1490, 6.6515])
lstm_seq_mae = np.array([2.4023, 2.6038, 2.6059])
transformer_seq_mae = np.array([2.7593, 2.4518, 3.2412])
naive_seq_mae = np.array([1.9096, 1.9045, 1.9795])

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.plot(seq_lengths, linear_seq_mae, marker="o", linewidth=2.0, label="Linear")
ax.plot(seq_lengths, lstm_seq_mae, marker="o", linewidth=2.0, label="LSTM")
ax.plot(seq_lengths, transformer_seq_mae, marker="o", linewidth=2.0, label="Transformer")
ax.plot(seq_lengths, naive_seq_mae, marker="o", linestyle="--", linewidth=1.8, label="Naive")
ax.set_xlabel("Sequence length")
ax.set_ylabel("MAE")
# ax.set_title("Sequence Length Ablation")
ax.set_xticks(seq_lengths)
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
ax.set_ylim(1.5, 7.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("seq_length_ablation_compact.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("Saved: feature_ablation_compact.png, seq_length_ablation_compact.png")
