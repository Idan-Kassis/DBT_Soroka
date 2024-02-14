import torch
import numpy as np
import os
from transformers import AutoImageProcessor, Swinv2Model, DefaultDataCollator
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ToPILImage, RandomResizedCrop, RandomVerticalFlip, RandomRotation, RandomHorizontalFlip
from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification
import pandas as pd
import random
import shutil
import torch.nn as nn
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import AutoFeatureExtractor, SwinForImageClassification
from transformers import AutoFeatureExtractor, SwinForImageClassification, CvtForImageClassification
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

pred_flag = True
evaluate_flag = True
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model_name = "Swin-base-DBT-1024-4-lr4"
device = 'cuda'

# data
# data
model_name_or_path = "microsoft/swin-base-patch4-window12-384-in22k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
transform = Compose([ToTensor(), normalize])

test_data_path = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-1024/Test"
test_dataset = ImageFolder(test_data_path, transform=transform)
batch_size = 4
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# model
labels = [0,1]
model_name_or_path = "microsoft/swin-base-patch4-window12-384-in22k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
model = SwinForImageClassification.from_pretrained(
    model_name_or_path,
    image_size=1024,
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

model = model.to(device)
save_path = str('/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model-1024/' + model_name + '/pytorch_model.bin')
model = model.to(device)
model.eval()
model.load_state_dict(torch.load(save_path))

def sig(x):
    return 1 / (1 + np.exp(-x))


def validate(model, loader):
    probabilities = []
    true_labels = []
    model.eval()  # set to eval mode to avoid batchnorm
    with torch.no_grad():  # avoid calculating gradients
        for images, labels in loader:
            img = images.to(device)
            p = sig(model(img).logits.cpu().numpy()[:, 1])
            probabilities.extend(p)
            true_labels.extend(labels)
    return probabilities, true_labels


# Train
probs, true = validate(model, test_dataloader)
neg_files = os.listdir(test_data_path+"/Negative")
pos_files = os.listdir(test_data_path+"/Positive")

print(np.array(true).reshape(len(true), ).shape)
print(np.array(probs).reshape(len(probs), ).shape)

results_train = pd.DataFrame({"Filename": neg_files + pos_files,
                              "true_labels": np.array(true).reshape(len(true), ),
                              "Probabilities": np.array(probs).reshape(len(probs), )})
results_train.to_csv(str("test_prediction-" + model_name + ".csv"), index=False)
print('Test prediction done!')

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

print('--------------------     Model - ', model_name, '     --------------------')
# Accuracy

df = pd.read_csv(str("test_prediction-" + model_name + ".csv"))

file_names = df["Filename"]

# get the subjects and view list
uniqe_filename = []
for name in file_names:
    if name[:12] not in uniqe_filename:
        uniqe_filename.append(name[:12])

probabilities = []
true_labels = []
ten_slice_prob = []
file_names = df["Filename"]
for name in uniqe_filename:
    res = file_names.str.contains(pat=name)
    relevant_probs = df["Probabilities"][res.values]
    y_true = df["true_labels"][res.values]
    probabilities.append(np.median(relevant_probs))
    true_labels.append(np.mean(y_true))

    maximum_prob = 0
    for idx in range(len(relevant_probs - 8)):
        current = np.mean(relevant_probs[idx:idx + 8])
        if current > maximum_prob:
            maximum_prob = current

    ten_slice_prob.append(maximum_prob)

probabilities = np.asarray(probabilities).reshape(len(probabilities), )
ten_slice_prob = np.asarray(ten_slice_prob).reshape(len(ten_slice_prob), )

true_labels = np.asarray(true_labels).reshape(len(true_labels), )
predictions = np.where(probabilities < 0.5, 0, 1).reshape(len(probabilities), )
ten_slice_prediction = np.where(ten_slice_prob < 0.5, 0, 1).reshape(len(ten_slice_prob), )

results = pd.DataFrame({"Filename": uniqe_filename,
                        "true_labels": true_labels,
                        "Predictions": ten_slice_prediction,
                        "Probabilities": ten_slice_prob})
results.to_csv("results_test_scan-based.csv", index=False)

# Metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

# Accuracy
predictions = ten_slice_prediction
probabilities = ten_slice_prob
acc = accuracy_score(true_labels, predictions)
print('Scan-based Accuracy: ', acc)

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
print(cm)

# sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sens = tp / (fn + tp)
spec = tn / (tn + fp)
print('Scan-based sensitivity - ', sens)
print('Scan-based Specificity - ', spec)

# AUC
auc = roc_auc_score(true_labels, probabilities, average=None)
print('Scan-based AUC - ', auc)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 4)))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('scan_based_ROC.jpg')
plt.close()
plt.clf()

# subject based
df = pd.read_csv("results_test_scan-based.csv")

file_names = df["Filename"]

# get the subjects and view list
uniqe_filename = []
for name in file_names:
    if name[:8] not in uniqe_filename:
        uniqe_filename.append(name[:8])

probabilities = []
true_labels = []
ten_slice_prob = []
file_names = df["Filename"]
for name in uniqe_filename:
    res = file_names.str.contains(pat=name)
    relevant_probs = df["Probabilities"][res.values]
    y_true = df["true_labels"][res.values]
    # probabilities.append(np.mean(relevant_probs))
    probabilities.append(np.mean(relevant_probs))
    true_labels.append(np.mean(y_true))
probabilities = np.asarray(probabilities).reshape(len(probabilities), )
true_labels = np.asarray(true_labels).reshape(len(true_labels), )
predictions = np.where(probabilities < 0.5, 0, 1).reshape(len(probabilities), )

acc = accuracy_score(true_labels, predictions)
print('Case-based Accuracy: ', acc)

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
print(cm)

# sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sens = tp / (fn + tp)
spec = tn / (tn + fp)
print('Case-based Sensitivity - ', sens)
print('Case-based Specificity - ', spec)

# AUC
auc = roc_auc_score(true_labels, probabilities, average=None)
print('Case-based AUC - ', auc)

fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 4)))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('scan_based_ROC.jpg')
plt.close()
plt.clf()