import os.path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
#from mlxtend.evaluate import DeLong
from sklearn.metrics import confusion_matrix
import scipy.stats
from scipy import stats
test_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Test"


def scan_based(file_names, probs, true):
    file_names = pd.Series(file_names)
    probs = pd.Series(probs)
    true = pd.Series(true)
    # get the subjects and view list
    uniqe_filename = []
    for name in file_names:
        if name[:12] not in uniqe_filename:
            uniqe_filename.append(name[:12])

    probabilities = []
    true_labels = []
    ten_slice_prob = []
    for name in uniqe_filename:
        res = file_names.str.contains(pat=name)
        relevant_probs =probs[res.values]
        y_true = true[res.values]
        probabilities.append(np.median(relevant_probs))
        true_labels.append(np.mean(y_true))
        maximum_prob = 0
        for idx in range(len(relevant_probs - 8)):
            current = np.mean(relevant_probs[idx:idx + 8])
            if current > maximum_prob:
                maximum_prob = current
        # maximum_prob = np.median(relevant_probs)
        ten_slice_prob.append(maximum_prob)

    ten_slice_prob = np.asarray(ten_slice_prob).reshape(len(ten_slice_prob), )
    true_labels = np.asarray(true_labels).reshape(len(true_labels), )
    pred = np.where(ten_slice_prob < 0.5, 0, 1).reshape(len(ten_slice_prob), )
    return uniqe_filename, true_labels, pred, ten_slice_prob

def case_based(file_names, probs, true):
    file_names = pd.Series(file_names)
    probs = pd.Series(probs)
    true = pd.Series(true)

    # get the subjects and view list
    uniqe_filename = []
    for name in file_names:
        if name[:8] not in uniqe_filename:
            uniqe_filename.append(name[:8])

    probabilities = []
    true_labels = []
    ten_slice_prob = []
    for name in uniqe_filename:
        res = file_names.str.contains(pat=name)
        relevant_probs = probs[res.values]
        y_true = true[res.values]
        probabilities.append(np.mean(relevant_probs))
        true_labels.append(np.mean(y_true))
    probabilities = np.asarray(probabilities).reshape(len(probabilities), )
    true_labels = np.asarray(true_labels).reshape(len(true_labels), )
    predictions = np.where(probabilities < 0.5, 0, 1).reshape(len(probabilities), )
    return true_labels, predictions, probabilities




# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
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


# Main
#model1 = "/workspace/DBT_US_Soroka/Codes/DBT/2D/baseline_CNN/results_test_scan-based.csv"
model1 = "/workspace/DBT_US_Soroka/Codes/DBT/2D/statistics/ssl-results/test_prediction-checkpoint.1.-ssl10.csv"
model2 = "/workspace/DBT_US_Soroka/Codes/DBT/2D/statistics/ssl-results/test_prediction-labels-10.csv"


df1 = pd.read_csv(model1)
probs1 = df1["Probabilities"]
true1 = df1["true_labels"]
names1 = df1["Filename"]
model_name1 = os.path.basename(model1)[16:-4]
scan_names, scans_true, _, scans_prob = scan_based(names1, probs1, true1)
case_true1, _, case_prob1 = case_based(scan_names, scans_prob, scans_true)

df2 = pd.read_csv(model2)
probs2 = df2["Probabilities"]
true2 = df2["true_labels"]
names2 = df2["Filename"]
model_name2 = os.path.basename(model2)[16:-4]
scan_names, scans_true, _, scans_prob = scan_based(names2, probs2, true2)
case_true2, _, case_prob2 = case_based(scan_names, scans_prob, scans_true)

print ('DeLongs Test - ', model_name1 ,'  &  ', model_name2)
p = delong_roc_test(case_true1, case_prob1, case_prob2)
print('P value - ', p)


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

alpha = .95
y_pred = case_prob2
y_true = case_true2

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
print('AUC COV:', auc_cov)
print('95% AUC CI:', ci)