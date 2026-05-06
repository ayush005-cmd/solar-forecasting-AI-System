"""
evaluate.py — SolarAI Model Evaluation & Visualization  (NEW)
==============================================================
Run AFTER train.py to generate two high-impact evaluation plots:

  1. Actual vs Predicted scatter  — per model, with perfect-fit diagonal
  2. Residual distribution         — histogram + KDE of prediction errors

Usage:
    python train.py          # builds the models
    python evaluate.py       # produces plots → model/plots/

Plots are saved as PNG (300 dpi) and can be shown during viva directly
from the file system OR embedded in a Jupyter notebook.

WHY THESE TWO PLOTS?
─────────────────────
• Actual vs Predicted:
  If points lie tightly along the y=x diagonal, the model predicts well
  across the full power range. Systematic curve = bias; fan shape = heteroscedasticity.

• Residual Distribution:
  Residuals should be approximately N(0, σ²) if the model has captured all signal.
  Left/right skew → systematic over/under-prediction.
  Heavy tails → occasional large errors the model misses.
  These are standard checks asked at ML viva examinations.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

PLOT_DIR = os.path.join("model", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load residual data from train.py output ───────────────────
DATA_FILE = os.path.join("model", "residual_data.json")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "model/residual_data.json not found.\n"
        "Please run  python train.py  first."
    )

with open(DATA_FILE) as f:
    pts = json.load(f)["points"]

actual     = np.array([p["actual"]         for p in pts])
svm_pred   = np.array([p["svmPred"]        for p in pts])
bag_pred   = np.array([p["baggingPred"]    for p in pts])
ts_pred    = np.array([p["timeSeriesPred"] for p in pts])
ens_pred   = np.array([p["ensemblePred"]   for p in pts])
svm_res    = np.array([p["svmResidual"]        for p in pts])
bag_res    = np.array([p["baggingResidual"]    for p in pts])
ts_res     = np.array([p["timeSeriesResidual"] for p in pts])
ens_res    = np.array([p["ensembleResidual"]   for p in pts])

# ── Colour palette ────────────────────────────────────────────
COLORS = {
    "SVM (SVR)":             "#22d3ee",
    "Bagging (RF)":          "#34d399",
    "Time-Series (Ridge)":   "#a78bfa",
    "Ensemble":              "#fbbf24",
}

# ─────────────────────────────────────────────────────────────
#  PLOT 1 — Actual vs Predicted (2×2 subplots, one per model)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Actual vs Predicted Power Output (Test Set)",
             fontsize=15, fontweight="bold", y=0.98)

models_avp = [
    ("SVM (SVR)",           svm_pred),
    ("Bagging (RF)",        bag_pred),
    ("Time-Series (Ridge)", ts_pred),
    ("Ensemble",            ens_pred),
]
axes_flat = axes.flatten()

for ax, (name, pred) in zip(axes_flat, models_avp):
    color = COLORS[name]
    ax.scatter(actual, pred, alpha=0.4, s=12, color=color, linewidths=0)

    # Perfect-fit diagonal
    lim_min = min(actual.min(), pred.min()) - 1
    lim_max = max(actual.max(), pred.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="white", linewidth=1.2, linestyle="--", alpha=0.8,
            label="Perfect fit (y = x)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # R² on the plot
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(actual - pred))

    ax.set_title(f"{name}\nR² = {r2:.4f}  |  MAE = {mae:.3f} kW",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual Power (kW)", fontsize=9)
    ax.set_ylabel("Predicted Power (kW)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

    # Dark background style
    ax.set_facecolor("#0f1624")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(facecolor="#1e293b", labelcolor="white", edgecolor="#334155")

fig.patch.set_facecolor("#0a0f1e")
plt.tight_layout(rect=[0, 0, 1, 0.97])

out1 = os.path.join(PLOT_DIR, "actual_vs_predicted.png")
plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[Plot 1] Saved: {out1}")

# ─────────────────────────────────────────────────────────────
#  PLOT 2 — Residual Distribution (histogram + KDE overlay)
# ─────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle("Residual Distribution — Prediction Errors (Test Set)\n"
              "Ideal: symmetric bell curve centred at 0",
              fontsize=14, fontweight="bold", y=0.99)

models_res = [
    ("SVM (SVR)",           svm_res),
    ("Bagging (RF)",        bag_res),
    ("Time-Series (Ridge)", ts_res),
    ("Ensemble",            ens_res),
]
axes2_flat = axes2.flatten()

for ax, (name, res) in zip(axes2_flat, models_res):
    color = COLORS[name]

    ax.hist(res, bins=40, color=color, alpha=0.55, density=True,
            edgecolor="none", label="Histogram")

    # KDE overlay
    try:
        kde = gaussian_kde(res, bw_method="scott")
        x_range = np.linspace(res.min(), res.max(), 300)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2, label="KDE")
    except Exception:
        pass

    # Zero-error reference line
    ax.axvline(0, color="white", linewidth=1.2, linestyle="--", alpha=0.9, label="Zero error")

    # Stats
    mean_res = res.mean()
    std_res  = res.std()
    ax.axvline(mean_res, color="#f87171", linewidth=1, linestyle=":",
               label=f"Mean = {mean_res:+.3f}")

    ax.set_title(f"{name}\nμ = {mean_res:+.3f} kW  |  σ = {std_res:.3f} kW",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual (Predicted − Actual)  [kW]", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    ax.set_facecolor("#0f1624")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(facecolor="#1e293b", labelcolor="white", edgecolor="#334155")

fig2.patch.set_facecolor("#0a0f1e")
plt.tight_layout(rect=[0, 0, 1, 0.95])

out2 = os.path.join(PLOT_DIR, "residual_distribution.png")
plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print(f"[Plot 2] Saved: {out2}")

print()
print("=" * 50)
print("  Evaluation plots generated!")
print(f"  → {out1}")
print(f"  → {out2}")
print("=" * 50)
print()
print("VIVA TALKING POINTS:")
print("  Plot 1 — Actual vs Predicted:")
print("    • Points close to y=x diagonal = low bias.")
print("    • Scatter should be symmetric, not funnel-shaped.")
print("    • Any systematic curve = non-linearity not captured.")
print()
print("  Plot 2 — Residual Distribution:")
print("    • Bell curve centred at 0 = unbiased model.")
print("    • Heavy right tail = model under-predicts peaks.")
print("    • σ (std of residuals) ≈ RMSE from metrics.json.")