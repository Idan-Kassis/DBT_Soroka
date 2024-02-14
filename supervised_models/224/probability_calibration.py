from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from datasets import load_dataset
import os
import random
from PIL import ImageDraw, ImageFont, Image
from transformers import ViTFeatureExtractor
import torch
import numpy as np
from datasets import load_metric
from transformers import ViTForImageClassification, AdamW
from transformers import Trainer
from transformers import TrainingArguments, EarlyStoppingCallback
import datasets
import transformers
import evaluate
from transformers import AutoImageProcessor, Swinv2Model, DefaultDataCollator
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ToPILImage, RandomResizedCrop, RandomVerticalFlip, RandomRotation, RandomHorizontalFlip
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import AutoFeatureExtractor, SwinForImageClassification, CvtForImageClassification
from torchvision import transforms
from transformers import LevitFeatureExtractor, LevitForImageClassificationWithTeacher
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def get_folders(path):
    folders = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            folders.append(item_path)
    return folders

folder_path = "/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model"
models_names = get_folders(folder_path)
print(models_names)
gpu_number = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
labels = [0, 1]

# Functions
def sig(x):
    return 1 / (1 + np.exp(-x))


def validate(model, loader):
    probabilities = []
    true_labels = []
    model.eval()  # set to eval mode to avoid batchnorm
    with torch.no_grad():  # avoid calculating gradients
        correct, total = 0, 0
        for images in loader:
            labels = images['labels']
            img = images["pixel_values"]
            probabilities.append(sig(model(img[None, :]).logits.numpy()[0][1]))
            true_labels.append(labels)
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)
    return probabilities, true_labels


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
            current = np.median(relevant_probs[idx:idx + 8])
            if current > maximum_prob:
                maximum_prob = current
        # maximum_prob = np.median(relevant_probs)
        ten_slice_prob.append(maximum_prob)

    ten_slice_prob = np.asarray(ten_slice_prob).reshape(len(ten_slice_prob), )
    true_labels = np.asarray(true_labels).reshape(len(true_labels), )
    pred = np.where(ten_slice_prob < 0.7, 0, 1).reshape(len(ten_slice_prob), )
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
    predictions = np.where(probabilities < 0.7, 0, 1).reshape(len(probabilities), )
    return true_labels, predictions, probabilities


def calculate_matrix(true, pred, prob, mode):
    print(mode)
    acc = accuracy_score(true, pred)
    print('Accuracy: ', acc)
    cm = confusion_matrix(true, pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (fn + tp)
    spec = tn / (tn + fp)
    print('Sensitivity - ', sens)
    print('Specificity - ', spec)
    # AUC
    auc = roc_auc_score(true, prob, average=None)
    print('AUC - ', auc)
# ------------------------------------ define validation data -------------------------------------------------------
dataset_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data"
dataset = load_dataset("imagefolder", data_files={"train": os.path.join(dataset_path, "Train/**"),
                                                  "test": os.path.join(dataset_path, "Test/**"),
                                                  "valid": os.path.join(dataset_path, "Val/**")},
                       drop_labels=False,
                       )

#normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
normalize = Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
_transforms = Compose([ToTensor(), normalize])
def transform(examples):
    examples["pixel_values"] = [_transforms(img) for img in examples["image"]]
    examples['labels'] = examples['label']
    del examples["image"]
    return examples
prepared_ds_val = dataset["valid"].with_transform(transform)

for full_model_name in models_names:
    model_name = os.path.basename(full_model_name)
    # ---------------------------------------- prepare the model --------------------------------------------------
    if model_name[:4] == 'BeiT':
        continue
        model_name_or_path = 'microsoft/beit-base-patch16-224-pt22k-ft22k'
        feature_extractor = BeitImageProcessor(model_name_or_path)
        model = BeitForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    elif model_name[:3] == 'CvT':
        continue
        model_name_or_path = 'microsoft/cvt-21'
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        model = CvtForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    elif model_name[:4] == 'DeiT':
        continue
        model_name_or_path = 'facebook/deit-base-patch16-224'
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    elif model_name[:4] == 'Swin' and model_name[-1] == '4':
        model_name_or_path = "microsoft/swin-base-patch4-window7-224-in22k"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        model = SwinForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    elif model_name[:3] == 'ViT':
        continue
        if model_name[12:14] == '32':
            model_name_or_path = 'google/vit-base-patch32-224-in21k'
        else:
            model_name_or_path = 'google/vit-base-patch16-224-in21k'
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    else:
        continue
    save_path = full_model_name + '/pytorch_model.bin'
    model.load_state_dict(torch.load(save_path))
    # ------------------------------------ predict validation data ------------------------------------------------
    y_prob_val, y_val = validate(model, prepared_ds_val)

    # -------------------------------------- Define & fit the calibrator ------------------------------------------
    calibrator = CalibratedClassifierCV(base_estimator=LogisticRegression(random_state=42), cv=5, method='sigmoid')
    calibrator.fit(y_prob_val.reshape(-1, 1), y_val)
    y_calib_prob_val = calibrator.predict_proba(y_prob_val.reshape(-1, 1))[:, 1]

    # ------------------------------------ load & calibrate test probabilities ------------------------------------
    test_csv = pd.read_csv("test_prediction-"+model_name+".csv")
    y_prob_test = test_csv['Probabilities'].to_numpy()
    y_test = test_csv['true_labels'].to_numpy()
    test_names = test_csv['Filename']
    y_calib_prob_test = calibrator.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]

    # ---------------------------------- scan & case based evaluation ---------------------------------------------
    print("-----------------------  ", model_name, "  -----------------------")
    scan_names, scans_true, scans_pred, scans_prob = scan_based(test_names, y_calib_prob_test, y_test)
    calculate_matrix(scans_true, scans_pred, scans_prob, 'Scan-based Evaluation - Calibrated')

    # case base
    case_true, case_pred, case_prob = case_based(scan_names, scans_prob, scans_true)
    calculate_matrix(case_true, case_pred, case_prob, 'Case-based Evaluation - Calibrated')

    # ------------------------------- visualization validation ------------------------------------
    # before callibration
    prob_true, prob_pred = calibration_curve(y_val, y_prob_val, n_bins=10)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.legend()

    prob_true, prob_pred = calibration_curve(y_val, y_calib_prob_val, n_bins=10)
    plt.plot(prob_pred, prob_true, "s-", label="calibrated Model")

    plt.legend()
    plt.savefig("calibration.png")
    plt.close()
    plt.clf()

