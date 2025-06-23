import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats
from sklearn.metrics import confusion_matrix
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    #return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    return scipy.stats.norm.sf(abs(z))*2


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_test(ground_truth, predictions_one, predictions_two):

    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def delong_roc_variance(ground_truth, predictions):
    predictions = np.asarray(predictions)
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def calculate_confidence_interval(y_true, y_pred):
    alpha = 0.95
    auc, auc_cov = delong_roc_variance(
        y_true,
        y_pred)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1

    print('AUC:', auc)
    print('95% CI:', auc - ci[0])


def calculate_confidence_interval_sens_spec(y_true, y_prob, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_bootstraps = 1000
    auc_values = np.zeros(n_bootstraps)
    sensitivity_values = np.zeros(n_bootstraps)
    specificity_values = np.zeros(n_bootstraps)


    for i in range(n_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        boot_y_true = y_true[indices]
        boot_y_prob = y_prob[indices]
        boot_y_pred = y_pred[indices]

        # AUC
        boot_auc = roc_auc_score(boot_y_true, boot_y_prob)
        auc_values[i] = boot_auc

        # sensitivity & specificity
        boot_cm = confusion_matrix(boot_y_true, boot_y_pred)
        tn, fp, fn, tp = boot_cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        sensitivity_values[i] = sensitivity
        specificity_values[i] = specificity

    # Calculate confidence interval
    sensitivity_ci_lower = np.percentile(sensitivity_values, 2.5)
    sensitivity_ci_upper = np.percentile(sensitivity_values, 97.5)
    specificity_ci_lower = np.percentile(specificity_values, 2.5)
    specificity_ci_upper = np.percentile(specificity_values, 97.5)

    print('Sensitivity - ', (sensitivity_ci_lower+sensitivity_ci_upper)/2,'   +-  ', sensitivity_ci_upper-(sensitivity_ci_lower+sensitivity_ci_upper)/2)
    print('Specificity - ', (specificity_ci_lower+specificity_ci_upper)/2,'   +-  ', specificity_ci_upper-(specificity_ci_lower+specificity_ci_upper)/2)

# ----------------------------------------- Main ------------------------------------------
swin_df = pd.read_csv("/Users/idankassis/Desktop/Thesis/medical paper - 1024/subgroups.csv")

# ===========================================================================================
# Biopsy
# only malignant
print("\n----- Biopsy - Malignant -----")
df = swin_df.copy()
df = df.drop(df[df['Biopsy_Result'] == "B"].index)
df = df.drop(df[df['Biopsy_Result'] == "EMPTY"].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)

# only benign
print("\n----- Biopsy - Benign -----")
df = swin_df.copy()
df = df.drop(df[df['Biopsy_Result'] == "M"].index)
df = df.drop(df[df['Biopsy_Result'] == "EMPTY"].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)

# ===========================================================================================
# Age
# age < 50
print("\n----- Age < 50  -----")
df = swin_df.copy()
df = df.drop(df[df['Age'] >= 50].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)

# age >= 50
print("\n----- Age >= 50  -----")
df = swin_df.copy()
df = df.drop(df[df['Age'] < 50].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)
# ===========================================================================================
# Size
def get_max_number(row):
    if pd.isna(row) or row == "":
        return np.nan

    numbers = []
    for item in row.split(","):
        sub_numbers = [int(num) for num in item.strip().split("X")]
        numbers.extend(sub_numbers)

    if len(numbers) > 0:
        return max(numbers)
    else:
        return np.nan


# size < 2
print("\n----- size < 2  -----")
df = swin_df.copy()
df['max_diameter'] = df['Tumor Size'].apply(get_max_number)
df = df.drop(df[df['max_diameter'] >= 20].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)

# size >= 2
print("\n----- size >= 2  -----")
df = swin_df.copy()
df['max_diameter'] = df['Tumor Size'].apply(get_max_number)
df = df.drop(df[df['max_diameter'] < 20].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)

# ===========================================================================================
# density
print("\n----- density a  -----")
df = swin_df.copy()
df = df.drop(df[df['BIRADS_Density'] != 'A'].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)

print("\n----- density B  -----")
df = swin_df.copy()
df = df.drop(df[df['BIRADS_Density'] != 'B'].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)

print("\n----- density C  -----")
df = swin_df.copy()
df = df.drop(df[df['BIRADS_Density'] != 'C'].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)

print("\n----- density D  -----")
df = swin_df.copy()
df = df.drop(df[df['BIRADS_Density'] != 'D'].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
preds = df["Predictions"]
calculate_confidence_interval(true_labels, probs)
calculate_confidence_interval_sens_spec(true_labels, probs, preds)