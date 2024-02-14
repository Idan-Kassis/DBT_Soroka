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

pred_flag = True
evaluate_flag = True
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
models_n = ["ViT-DBT-224-16-lr5", "ViT-DBT-224-32-lr4", "ViT-DBT-224-16-lr4",
            "BeiT-DBT-224-16-lr3", "BeiT-DBT-224-16-lr5",
            "Swin-DBT-224-4-lr5",
            "CvT-DBT-224-16-lr3","CvT-DBT-224-16-lr4","CvT-DBT-224-16-lr5",
            'DeiT-DBT-224-16-lr3', 'DeiT-DBT-224-16-lr4', 'DeiT-DBT-224-16-lr5']

# model
labels = [0, 1]
dataset = load_dataset("imagefolder", data_files={"test": "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Test/**"},
                        drop_labels=False,
)

for index, model_name in enumerate(models_n):
    if index == 0:
        model_name_or_path = 'google/vit-base-patch16-224-in21k'
        img_shape = 224
        continue
    elif index == 1:
        model_name_or_path = 'google/vit-base-patch32-224-in21k'
        img_shape = 224
        continue
    elif index == 2:
        model_name_or_path = 'google/vit-base-patch16-224-in21k'
        img_shape = 224
        continue
    elif index == 3 or index ==4:
        model_name_or_path = 'microsoft/beit-base-patch16-224-pt22k-ft22k'
        img_shape = 224
        continue
    elif index == 5:
        model_name_or_path = "microsoft/swin-base-patch4-window7-224-in22k"
        img_shape = 224
        continue
    elif index > 5 and index < 9:
        model_name_or_path = 'microsoft/cvt-21'
        img_shape = 224
        continue
    elif index >= 9 and index < 12:
        model_name_or_path = 'facebook/deit-base-patch16-224'
        img_shape = 224
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )



    if index < 3:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    elif index == 3 or index == 4:
        feature_extractor = BeitImageProcessor(model_name_or_path)
        model = BeitForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    elif index == 5:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        model = SwinForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    elif index > 5 and index<9:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        model = CvtForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )

    if pred_flag:
        normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        _transforms = Compose([ToTensor(), normalize])
        def transform(examples):
            examples["pixel_values"] = [_transforms(img) for img in examples["image"]]
            examples['labels'] = examples['label']
            del examples["image"]
            return examples
        prepared_ds = dataset["test"].with_transform(transform)

        save_path = str('/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model/'+model_name+'/pytorch_model.bin')
        print ('--------------------     Model - ', model_name, '     --------------------')
        model.load_state_dict(torch.load(save_path))

        def sig(x):
            return 1/(1 + np.exp(-x))


        def validate(model, loader):
            probabilities = []
            true_labels = []
            model.eval()  # set to eval mode to avoid batchnorm
            with torch.no_grad():  # avoid calculating gradients
                correct, total = 0, 0
                for images in loader:
                    labels = images['labels']
                    img = images["pixel_values"]
                    probabilities.append(sig(model(img[None,:]).logits.numpy()[0][1]))
                    true_labels.append(labels)
            return probabilities, true_labels

        # Train
        probs, true = validate(model, prepared_ds)
        neg_files = os.listdir("/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Test/Negative")
        pos_files = os.listdir("/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Test/Positive")


        results_train = pd.DataFrame({"Filename": neg_files+pos_files,
                                "true_labels": np.array(true).reshape(len(true), ),
                                "Probabilities": np.array(probs).reshape(len(probs), )})
        results_train.to_csv(str("test_prediction-"+model_name+".csv"), index=False)
        print('Test prediction done!')


    from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

    print('--------------------     Model - ', model_name, '     --------------------')
    # Accuracy

    df = pd.read_csv(str("test_prediction-"+model_name+".csv"))

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
        for idx in range(len(relevant_probs-8)):
            current = np.mean(relevant_probs[idx:idx+8])
            if current>maximum_prob:
                maximum_prob = current

        ten_slice_prob.append(maximum_prob)

    probabilities = np.asarray(probabilities).reshape(len(probabilities), )
    ten_slice_prob = np.asarray(ten_slice_prob).reshape(len(ten_slice_prob), )

    true_labels = np.asarray(true_labels).reshape(len(true_labels), )
    predictions = np.where(probabilities < 0.62, 0, 1).reshape(len(probabilities), )
    ten_slice_prediction = np.where(ten_slice_prob < 0.62, 0, 1).reshape(len(ten_slice_prob), )


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
        #probabilities.append(np.mean(relevant_probs))
        probabilities.append(np.mean(relevant_probs))
        true_labels.append(np.mean(y_true))
    probabilities = np.asarray(probabilities).reshape(len(probabilities), )
    true_labels = np.asarray(true_labels).reshape(len(true_labels), )
    predictions = np.where(probabilities < 0.62, 0, 1).reshape(len(probabilities), )

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

