import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats
age_flag = False

if age_flag:
    swin_csv = pd.read_csv("swin_subgroups.csv")
    metadata_df = pd.read_excel("/workspace/DBT_US_Soroka/Codes/DBT/Soroka_DBT_Metadata_Dicom.xlsx")

    # Column in df1 to search for elements
    column_to_search = 'Filename'

    # Column in df2 to check for existence
    column_to_check = 'Name'
    age_list = []
    # Iterate over each element in the column_to_search of df1
    for element in swin_csv[column_to_search]:
        # Check if the element exists in the column_to_check of df2
        if element[:-2] in metadata_df[column_to_check].values:
            # Get the value from another column in the same row
            age = metadata_df.loc[metadata_df[column_to_check] == element[:-2], 'Age'].values[0]
            age_list.append(int(age))

    swin_csv['Age'] = age_list
    swin_csv.to_csv('swin_subgroups.csv', index=False)


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
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

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
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

# ----------------------------------------- Main ------------------------------------------
swin_df = pd.read_csv("swin_ssl_subgroups.csv")

# ===========================================================================================
# Biopsy
# only malignant
print("----- Biopsy - Malignant -----")
df = swin_df.copy()
df = df.drop(df[df['Biopsy'] == "B"].index)
df = df.drop(df[df['Biopsy'] == "EMPTY"].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
calculate_confidence_interval(true_labels, probs)

# only benign
print("----- Biopsy - Benign -----")
df = swin_df.copy()
df = df.drop(df[df['Biopsy'] == "M"].index)
df = df.drop(df[df['Biopsy'] == "EMPTY"].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
calculate_confidence_interval(true_labels, probs)


# ===========================================================================================
# Age
# age < 50
print("----- Age < 50  -----")
df = swin_df.copy()
df = df.drop(df[df['Age'] >= 50].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
calculate_confidence_interval(true_labels, probs)

# age >= 50
print("----- Age >= 50  -----")
df = swin_df.copy()
df = df.drop(df[df['Age'] < 50].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
calculate_confidence_interval(true_labels, probs)

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
print("----- size < 2  -----")
df = swin_df.copy()
df['max_diameter'] = df['Size'].apply(get_max_number)
df = df.drop(df[df['max_diameter'] >= 20].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
calculate_confidence_interval(true_labels, probs)

# size >= 2
print("----- size >= 2  -----")
df = swin_df.copy()
df['max_diameter'] = df['Size'].apply(get_max_number)
df = df.drop(df[df['max_diameter'] < 20].index)
df = df.reset_index(drop=True)
true_labels = df["true_labels"]
probs = df["Probabilities"]
calculate_confidence_interval(true_labels, probs)

