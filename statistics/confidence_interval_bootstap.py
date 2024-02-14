import os.path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
#from mlxtend.evaluate import DeLong
from sklearn.metrics import confusion_matrix

test_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Test"

def calculate_confidence_interval(y_true, y_prob, y_pred, model_n, evaluation_mode, dependant_flag):
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
        test_names = os.listdir(os.path.join(test_path,'Negative')) + os.listdir(os.path.join(test_path,'Positive'))
        uniqe_filename = []
        for name in test_names:
            if name[:12] not in uniqe_filename:
                uniqe_filename.append(name[:12])
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
    print(model_n)
    print(evaluation_mode)
    print('AUC - ',(ci_lower_auc+ci_upper_auc)/2 ,'   +-  ', ci_upper_auc-(ci_lower_auc+ci_upper_auc)/2)
    print("AUC Confidence Interval (95%): [{:.4f}, {:.4f}]".format(ci_lower_auc, ci_upper_auc))
    print('Sensitivity - ', (sensitivity_ci_lower+sensitivity_ci_upper)/2,'   +-  ', sensitivity_ci_upper-(sensitivity_ci_lower+sensitivity_ci_upper)/2)
    print("Sensitivity Confidence Interval (95%): [{:.4f}, {:.4f}]".format(sensitivity_ci_lower, sensitivity_ci_upper))
    print('Specificity - ', (specificity_ci_lower+specificity_ci_upper)/2,'   +-  ', specificity_ci_upper-(specificity_ci_lower+specificity_ci_upper)/2)
    print("Specificity Confidence Interval (95%): [{:.4f}, {:.4f}]".format(specificity_ci_lower, specificity_ci_upper))


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
        sorted_list = sorted(zip(list(df["Filename"][res.values]), list(df["Probabilities"][res.values])),
                             key=lambda x: int(x[0].split("_")[-1].split(".")[0]))
        sorted_list_probs = [item[1] for item in sorted_list]
        relevant_probs = np.array(sorted_list_probs)
        #relevant_probs =probs[res.values]
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
    pred = np.where(ten_slice_prob < 0.0012, 0, 1).reshape(len(ten_slice_prob), )
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
    predictions = np.where(probabilities < 0.0012, 0, 1).reshape(len(probabilities), )
    return true_labels, predictions, probabilities

#model_csv_res = ["/workspace/DBT_US_Soroka/Codes/DBT/2D/pretrain_ssl/mt-original/test_prediction-student_model-ssl4-lr5-loss-MSE.pth.csv"]
#model_csv_res = ["/workspace/DBT_US_Soroka/Codes/DBT/2D/self_supervised_mean-teacher/ssl-weight-384-experiment/pseudo_exp/test_prediction-student_model-ssl1-lr8-loss-CE-PSEUDO-EXT.pth.csv"]
model_csv_res = ["/workspace/DBT_US_Soroka/Codes/DBT/2D/statistics/ssl-results/test_prediction-checkpoint.4.ssl1.csv"]

#"/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model/test_prediction-ViT-DBT-224-16-lr4.csv"
#"/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model/test_prediction-Swin-DBT-224-4-lr4.csv"
#"/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model-384/test_prediction-Swin-base-DBT-384-4-lr4.csv",
#"/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model-1024/test_prediction-Swin-base-DBT-1024-4-lr4.csv"


for csv_name in model_csv_res:
    df = pd.read_csv(csv_name)
    probs = df["Probabilities"]
    true = df["true_labels"]
    names = df["Filename"]
    model_name = os.path.basename(csv_name)[16:-4]

    #scan_names, scans_true, scans_pred, scans_prob = scan_based(names, probs, true)
    #calculate_confidence_interval(scans_true, scans_prob, scans_pred, model_name, 'Scan-based', True)

    case_true, case_pred, case_prob = case_based(names, probs, true)
    calculate_confidence_interval(case_true, case_prob, case_pred, model_name, 'Case-based', False)
