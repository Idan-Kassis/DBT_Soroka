import os.path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
#from mlxtend.evaluate import DeLong
from sklearn.metrics import confusion_matrix

test_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Test"

def calculate_confidence_interval(y_true, y_prob, y_pred, evaluation_mode, dependant_flag, uniqe_filename=None):
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    n_bootstraps = 1000
    auc_values = np.zeros(n_bootstraps)
    sensitivity_values = np.zeros(n_bootstraps)
    specificity_values = np.zeros(n_bootstraps)

    # Perform bootstrap sampling

    if dependant_flag:
        for i in range(n_bootstraps):
            indices = []
            for j in range(len(y_true)):
                if len(indices) == len(y_true):
                    break
                idx = int(np.random.choice(len(y_true), 1, replace=True))
                indices.append(idx)
                if len(indices) == len(y_true):
                    break
                for add_idx, item in enumerate(uniqe_filename):
                    if uniqe_filename[idx][:7] in item:
                        indices.append(add_idx)
                        if len(indices) == len(y_true):
                            break
            indices = np.array(indices)
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
    else:
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
    ci_lower_auc = np.percentile(auc_values, 2.5)
    ci_upper_auc = np.percentile(auc_values, 97.5)
    sensitivity_ci_lower = np.percentile(sensitivity_values, 2.5)
    sensitivity_ci_upper = np.percentile(sensitivity_values, 97.5)
    specificity_ci_lower = np.percentile(specificity_values, 2.5)
    specificity_ci_upper = np.percentile(specificity_values, 97.5)

    print('===========================================================================================================')
    print(evaluation_mode)
    print('AUC - ',(ci_lower_auc+ci_upper_auc)/2 ,'   +-  ', ci_upper_auc-(ci_lower_auc+ci_upper_auc)/2)
    print("AUC Confidence Interval (95%): [{:.4f}, {:.4f}]".format(ci_lower_auc, ci_upper_auc))
    print('Sensitivity - ', (sensitivity_ci_lower+sensitivity_ci_upper)/2,'   +-  ', sensitivity_ci_upper-(sensitivity_ci_lower+sensitivity_ci_upper)/2)
    print("Sensitivity Confidence Interval (95%): [{:.4f}, {:.4f}]".format(sensitivity_ci_lower, sensitivity_ci_upper))
    print('Specificity - ', (specificity_ci_lower+specificity_ci_upper)/2,'   +-  ', specificity_ci_upper-(specificity_ci_lower+specificity_ci_upper)/2)
    print("Specificity Confidence Interval (95%): [{:.4f}, {:.4f}]".format(specificity_ci_lower, specificity_ci_upper))



#scan based
#df = pd.read_csv("/Users/idankassis/Desktop/Thesis/codes/pythonProject1/predictions/1024-res/results_test_scan-based_1024.csv")
#scans_prob = df["Probabilities"]
#scans_true = df["true_labels"]
#scans_pred = df["Predictions"]
#names = list(df["Filename"])
#calculate_confidence_interval(scans_true, scans_prob, scans_pred, 'Scan-based', True, names)

df = pd.read_csv("/Users/idankassis/Desktop/Thesis/codes/pythonProject1/predictions/1024-res/results_test_case-based_1024_biopsy.csv")
case_prob = df["Probabilities"]
case_true = df["true_labels"]
case_pred = df["Predictions"]

calculate_confidence_interval(case_true, case_prob, case_pred, 'Case-based', False)
