import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


# def run_significance_test(x, y):
#     """
#     Paired Wilcoxon signed-rank test.
#     Returns: statistic, p-value
#     """
#     stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
#     return stat, p


# def compute_effect_size(x, y):
#     """
#     Effect size for paired data.
#     Returns:
#       - median_shift: median(y - x)
#       - rank-biserial correlation
#     """
#     diff = y - x
#     median_shift = float(np.median(diff))

#     nonzero = diff != 0
#     d = diff[nonzero]

#     if len(d) == 0:
#         return median_shift, 0.0

#     ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
#     w_pos = float(np.sum(ranks[d > 0]))
#     n = len(d)
#     max_w = n * (n + 1) / 2.0

#     rbc = (w_pos - (max_w - w_pos)) / max_w
#     return median_shift, float(rbc)


# def load_column(csv_path, column_1_name, column_2_name):
#     df = pd.read_csv(csv_path)

#     if column_1_name not in df.columns or column_2_name not in df.columns:
#         raise ValueError(
#             f"One or both columns not found in {csv_path}. "
#             f"Available columns: {list(df.columns)}"
#         )

#     col1 = df[column_1_name].astype(float).to_numpy()
#     col2 = df[column_2_name].astype(float).to_numpy()

#     return col1, col2


# def main():
#     # ===== HARD-CODED PATHS =====
#     dataset_name = 'covid-19'  # 'TALLIP','liar_2','liar_6', 'kaggle', 'covid-19', 'election_2024
#     paraphraser = 'gemini'

#     column_name_1 = "Preserved_sentiment"  # change if needed
#     column_name_2 = "Human_sentiment"  # change if needed
#     dataset = f"results/sentiment/{dataset_name}_{paraphraser}_vader_scores.csv"

#     # Load data
#     data_1, data_2 = load_column(dataset, column_name_1,column_name_2)
#     print(data_1)
#     print(data_2)

#     # Safety check for paired test
#     if len(data_1) != len(data_2):
#         raise ValueError(
#             f"Row mismatch: {len(data_1)} vs {len(data_2)}. "
#             "Paired tests require aligned rows."
#         )

#     # Run tests
#     stat, p = run_significance_test(data_1, data_2)
#     median_shift, rbc = compute_effect_size(data_1, data_2)

#     # Output
#     print("=== Paired Sentiment Shift Test ===")
#     print(f"Wilcoxon statistic: {stat:.4f}")
#     print(f"p-value: {p:.6g}")
#     print(f"Median sentiment shift ({column_name_1} − {column_name_2}): {median_shift:.4f}")
#     print(f"Rank-biserial correlation: {rbc:.4f}")

#     if p < 0.05:
#         print("Conclusion: Statistically significant sentiment shift.")
#     else:
#         print("Conclusion: No statistically significant sentiment shift.")


# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttost_paired

# Load data
df = pd.read_csv("results/sentiment/liar_gemini_vader_scores.csv")

# human = df["Preserved_sentiment"].values
# paraphrased = df["Human_sentiment"].values

# # Equivalence margin
# delta = 0.10  # justify this in your paper

# # TOST paired test
# tost_pvalue = ttost_paired(
#     paraphrased,
#     human,
#     low=-delta,
#     upp=delta
# )[0]

# print(f"TOST p-value: {tost_pvalue:}")

# if tost_pvalue < 0.05:
#     print("Conclusion: Sentiment is statistically equivalent (no meaningful shift).")
# else:
#     print("Conclusion: Cannot conclude sentiment equivalence.")


import numpy as np
from scipy import stats

human = df["Human_sentiment"]
other = df["Positive_sentiment"]

diff =  human - other

t_stat, p_two = stats.ttest_rel(human, other, nan_policy="omit")
# convert to one-sided p-value for "greater"
p_one = p_two/2 if t_stat > 0 else 1 - p_two/2

print("Paired t-test p:", p_two)
print("Paired t-test t:", t_stat)
print("One-sided p (pos > human):", p_one)
print(f"One-sided p-value: {p_one:.3e}")
print("Mean diff:", np.mean(diff))
