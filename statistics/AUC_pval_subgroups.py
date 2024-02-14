
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
from sklearn.metrics import roc_auc_score

res_df = pd.read_csv("swin_ssl_subgroups.csv")
n_bootstraps = 10
auc_benign = np.zeros(n_bootstraps)
auc_malignant = np.zeros(n_bootstraps)

# benign
df = res_df.copy()
df = df.drop(df[df['Biopsy'] == "M"].index)
df = df.drop(df[df['Biopsy'] == "EMPTY"].index)
df = df.reset_index(drop=True)
y_benign = df["true_labels"]
prob_benign = df["Probabilities"]

# malignant
df = res_df.copy()
df = df.drop(df[df['Biopsy'] == "B"].index)
df = df.drop(df[df['Biopsy'] == "EMPTY"].index)
df = df.reset_index(drop=True)
y_mal = df["true_labels"]
prob_mal = df["Probabilities"]


# benign
for i in range(n_bootstraps):
    indices = np.random.choice(len(y_benign), len(y_benign), replace=True)
    boot_y_true = y_benign[indices]
    boot_y_prob = prob_benign[indices]
    # AUC
    boot_auc = roc_auc_score(boot_y_true, boot_y_prob)
    auc_benign[i] = boot_auc

# Malignant
for i in range(n_bootstraps):
    indices = np.random.choice(len(y_mal), len(y_mal), replace=True)
    boot_y_true = y_mal[indices]
    boot_y_prob = prob_mal[indices]
    # AUC
    boot_auc = roc_auc_score(boot_y_true, boot_y_prob)
    auc_malignant[i] = boot_auc

statistic, p_value = mannwhitneyu(auc_benign, auc_malignant, method='asymptotic')





import scipy.stats as stats

def calculate_p_value(auc_1, ci_1, auc_2, ci_2):
    # Step 1: Check for overlap
    #if ci_1[1] >= ci_2[0] and ci_1[0] <= ci_2[1]:
    #    print("The confidence intervals overlap. Cannot calculate p-value.")
    #    return None

    # Step 2: Calculate z-score
    z_score = (auc_1 - auc_2) / ((ci_1[1] - ci_1[0])**2 / (4 * 1.96**2) + (ci_2[1] - ci_2[0])**2 / (4 * 1.96**2))**0.5

    # Step 3: Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return p_value

# Example usage:
auc_1 = 0.926
ci1 = 0.035
ci_1 = (auc_1-ci1, auc_1+ci1)
auc_2 = 0.942
ci2 = 0.034
ci_2 = (auc_2-ci2, auc_2+ci2)

p_value = calculate_p_value(auc_1, ci_1, auc_2, ci_2)
print("P-value:", p_value)


