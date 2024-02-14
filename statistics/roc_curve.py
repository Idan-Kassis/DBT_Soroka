import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import sem
import pandas as pd
from scipy import stats


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
    return auc - ci[0]


# Case Base ROC Curve

# Renset - 224
model = "/workspace/DBT_US_Soroka/Codes/DBT/2D/baseline_CNN/results_test_scan-based.csv"
df = pd.read_csv(model)
probs = df["Probabilities"]
true = df["true_labels"]
names = df["Filename"]
true1, _, prob1 = case_based(names, probs, true)

# ViT- 224
model = "/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model/test_prediction-ViT-DBT-224-16-lr4.csv"
df = pd.read_csv(model)
probs = df["Probabilities"]
true = df["true_labels"]
names = df["Filename"]
scan_names, scans_true, _, scans_prob = scan_based(names, probs, true)
true2, _, prob2 = case_based(scan_names, scans_prob, scans_true)

# Swin - 224
model = "/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model/test_prediction-Swin-DBT-224-4-lr4.csv"
df = pd.read_csv(model)
probs = df["Probabilities"]
true = df["true_labels"]
names = df["Filename"]
scan_names, scans_true, _, scans_prob = scan_based(names, probs, true)
true3, _, prob3 = case_based(scan_names, scans_prob, scans_true)

# Swin 384
model = "/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model-384/test_prediction-Swin-base-DBT-384-4-lr4.csv"
df = pd.read_csv(model)
probs = df["Probabilities"]
true = df["true_labels"]
names = df["Filename"]
scan_names, scans_true, _, scans_prob = scan_based(names, probs, true)
true4, _, prob4 = case_based(scan_names, scans_prob, scans_true)

# Swin 1024
model = "/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model-1024/test_prediction-Swin-base-DBT-1024-4-lr4.csv"
df = pd.read_csv(model)
probs = df["Probabilities"]
true = df["true_labels"]
names = df["Filename"]
scan_names, scans_true, _, scans_prob = scan_based(names, probs, true)
true5, _, prob5 = case_based(scan_names, scans_prob, scans_true)


# Compute ROC curve and AUC for each model
fpr_model1, tpr_model1, _ = roc_curve(true1, prob1)
roc_auc_model1 = auc(fpr_model1, tpr_model1)

fpr_model2, tpr_model2, _ = roc_curve(true2, prob2)
roc_auc_model2 = auc(fpr_model2, tpr_model2)

fpr_model3, tpr_model3, _ = roc_curve(true3, prob3)
roc_auc_model3 = auc(fpr_model3, tpr_model3)

fpr_model4, tpr_model4, _ = roc_curve(true4, prob4)
roc_auc_model4 = auc(fpr_model4, tpr_model4)

fpr_model5, tpr_model5, _ = roc_curve(true5, prob5)
roc_auc_model5 = auc(fpr_model5, tpr_model5)

# Compute confidence intervals for each model
ci1 = calculate_confidence_interval(true1, prob1)
ci2 = calculate_confidence_interval(true2, prob2)
ci3 = calculate_confidence_interval(true3, prob3)
ci4 = calculate_confidence_interval(true4, prob4)
ci5 = calculate_confidence_interval(true5, prob5)

fig = plt.figure(figsize=(10, 8), dpi=300)

# Plot ROC curve and confidence intervals for each model
plt.plot(fpr_model1, tpr_model1, label='ResNet101 - 224 (AUC = %0.2f)' % roc_auc_model1, linestyle = '--')
#plt.fill_between(fpr_model1, tpr_model1-ci1, tpr_model1+ci1, alpha=0.3)

plt.plot(fpr_model2, tpr_model2, label='ViT - 224 (AUC = %0.2f)' % roc_auc_model2, linestyle = '-.')
#plt.fill_between(fpr_model2, tpr_model2-ci2, tpr_model2+ci2, alpha=0.3)

plt.plot(fpr_model3, tpr_model3, label='Swin - 224 (AUC = %0.2f)' % roc_auc_model3, linestyle = ':')
#plt.fill_between(fpr_model3, tpr_model3-ci3, tpr_model3+ci3, alpha=0.3)

plt.plot(fpr_model4, tpr_model4, label='Swin - 384 (AUC = %0.2f)' % roc_auc_model4, linestyle = '-')
#plt.fill_between(fpr_model4, tpr_model4-ci4, tpr_model4+ci4, alpha=0.3)

plt.plot(fpr_model5, tpr_model5, label='Swin - 1024 (AUC = %0.2f)' % roc_auc_model5, linestyle = '--')
#plt.fill_between(fpr_model5, tpr_model5-ci5, tpr_model5+ci5, alpha=0.3)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('tight')
#plt.title('ROC Curve of Supervised Models')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.legend(loc=4,fontsize=14)
plt.savefig('ROC.jpg')
plt.close()
plt.clf()



