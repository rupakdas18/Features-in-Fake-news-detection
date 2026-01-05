


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import variables

# =========================
# 1) Load your CSV
# =========================

variables.dataset_name = 'liar' # 'TALLIP''liar_2','liar_6', 'kaggle', 'covid-19', 'election_2024
CSV_PATH = f"results/{variables.dataset_name}_all_results.csv"   # <-- change if needed
df = pd.read_csv(CSV_PATH)

# =========================
# 2) Settings
# =========================
MODELS = ["CNN", "LSTM", "GPT", "BERT", "Llama"]
CONDS  = ["Human", "Preserved", "Positive", "Negative"]

# Column names in YOUR csv
TRUE_COL = "True_label"
PRED_COL = {
    "Human":   {m: f"{m}_human"   for m in MODELS},
    "Preserved": {m: f"{m}_preserved" for m in MODELS},
    "Positive":{m: f"{m}_positive"for m in MODELS},
    "Negative":{m: f"{m}_negative"for m in MODELS},
}

# Choose F1 averaging:
# - For binary: "binary"
# - For multi-class: "macro" (harder) or "weighted" (common)
F1_AVG = "weighted"

N_BOOT = 2000
SEED = 42

# Vertical offsets so conditions don’t overlap per model row
OFFSETS = {"Human": -0.18, "Preserved": -0.06, "Positive": 0.06, "Negative": 0.18}

# =========================
# 3) Bootstrap CI function
# =========================
def bootstrap_f1_ci(y_true, y_pred, average="weighted", n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    scores = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)  # sample with replacement
        scores[b] = f1_score(y_true[idx], y_pred[idx], average=average)

    mean = scores.mean()
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return mean, lo, hi

# =========================
# 4) Compute summary table
#    (drops NaNs per model/condition automatically)
# =========================
rows = []
for m in MODELS:
    for c in CONDS:
        col = PRED_COL[c][m]
        sub = df[[TRUE_COL, col]].dropna()   # important: your csv has NaNs in many rows
        if len(sub) == 0:
            continue

        mean, lo, hi = bootstrap_f1_ci(
            sub[TRUE_COL].values,
            sub[col].values,
            average=F1_AVG,
            n_boot=N_BOOT,
            seed=SEED
        )
        rows.append({"Model": m, "Condition": c, "F1": mean, "Lo": lo, "Hi": hi, "N": len(sub)})

summary = pd.DataFrame(rows)

# =========================
# 5) Plot (dot + horizontal 95% CI)
# =========================

COND_COLORS = {
    "Human":    "#1f77b4",  # blue
    "Preserved":  "#2ca02c",  # green
    "Positive": "#d62728",  # red
    "Negative": "#9467bd",  # purple
}


fig, ax = plt.subplots(figsize=(12, 6))

y_base = {m: i for i, m in enumerate(MODELS)}

for cond in CONDS:
    subc = summary[summary["Condition"] == cond]
    for _, r in subc.iterrows():
        y = y_base[r["Model"]] + OFFSETS[cond]
        x = r["F1"]
        xerr = np.array([[x - r["Lo"]], [r["Hi"] - x]])

        ax.errorbar(
            x, y,
            xerr=xerr,
            fmt="o",
            capsize=5,
            elinewidth=2,
            color=COND_COLORS[cond],        # <-- FIX
            markeredgecolor="black",
            label=cond if r["Model"] == MODELS[0] else None
)
        # numeric label next to point (optional)
        ax.text(x, y + 0.03, f"{x:.3f}", fontsize=9)

ax.set_yticks([y_base[m] for m in MODELS])
ax.set_yticklabels(MODELS)
ax.set_xlabel("F1 score", fontsize=20)
ax.set_ylabel("Classifiers", fontsize=20)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.grid(True, axis="x", linestyle="--", alpha=0.35)
ax.grid(True, axis="y", linestyle="--", alpha=0.15)
ax.legend(title="Sentiment", loc="upper left",
          fontsize=16, title_fontsize=18)

plt.tight_layout()

# Save or show
plt.savefig(f"results/{variables.dataset_name}_f1_ci_by_model_and_sentiment.pdf", dpi=300, bbox_inches="tight")
plt.show()

print(summary.sort_values(["Model", "Condition"]))
