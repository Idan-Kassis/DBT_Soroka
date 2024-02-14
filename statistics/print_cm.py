import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import sem
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score


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
    pred = np.where(ten_slice_prob < 0.55, 0, 1).reshape(len(ten_slice_prob), )
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
    predictions = np.where(probabilities < 0.55, 0, 1).reshape(len(probabilities), )
    return true_labels, predictions, probabilities


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
model = "/workspace/DBT_US_Soroka/Codes/DBT/2D/pretrain_ssl/mt-original/test_prediction-student_model-ssl4-lr5-loss-MSE.pth.csv"
df = pd.read_csv(model)
probs = df["Probabilities"]
true = df["true_labels"]
names = df["Filename"]
scan_names, scans_true, _, scans_prob = scan_based(names, probs, true)
true5, _, prob5 = case_based(scan_names, scans_prob, scans_true)


# Swin 384 - biopsy
model = "/workspace/DBT_US_Soroka/Codes/DBT/2D/statistics/results_test_case-based_supervised-swin_biopsybased.csv"
df = pd.read_csv(model)
prob6 = df["Probabilities"]
true6 = df["true_labels"]


# confusion matrix
from sklearn.metrics import confusion_matrix
true = true5
prob = prob5
pred = np.where(prob < 0.55, 0, 1).reshape(len(prob), )
cm = confusion_matrix(true, pred)
tn, fp, fn, tp = cm.ravel()
sens = tp / (fn + tp)
spec = tn / (tn + fp)
print('Sensitivity - ', sens)
print('Specificity - ', spec)
print(cm)
