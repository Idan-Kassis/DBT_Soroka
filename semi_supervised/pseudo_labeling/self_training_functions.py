import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import random
from PIL import ImageDraw, ImageFont, Image
#from transformers import ViTImageProcessor
import torch
from datasets import load_metric
from transformers import ViTForImageClassification, AdamW
from transformers import Trainer
from transformers import TrainingArguments, EarlyStoppingCallback
import datasets
import transformers
import evaluate
from transformers import AutoImageProcessor, Swinv2Model, DefaultDataCollator
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ToPILImage, RandomResizedCrop, RandomVerticalFlip, RandomRotation, RandomHorizontalFlip, ColorJitter
from torch.utils.data import DataLoader
import shutil
from self_training_model import collate_fn, compute_metrics, CustomTrainer
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from transformers import AutoFeatureExtractor, SwinForImageClassification, CvtForImageClassification


def train_data_transform(examples):
    img_shape=224
    normalize = Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    #_transforms = Compose([Resize((img_shape, img_shape)), RandomVerticalFlip(0.5), RandomRotation(10), RandomHorizontalFlip(0.5), ToTensor(),normalize])
    _transforms = Compose([ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                RandomVerticalFlip(0.5),
                                RandomRotation(10),
                                RandomHorizontalFlip(0.5),
                                ToTensor(),
                                normalize])
    examples["pixel_values"] = [_transforms(img) for img in examples["image"]]
    examples['labels'] = examples['label']
    del examples["image"]
    return examples

def data_transform(examples):
    img_shape=224
    normalize = Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    #_transforms = Compose([Resize((img_shape, img_shape)), ToTensor(), normalize])
    _transforms = Compose([ToTensor(), normalize])
    examples["pixel_values"] = [_transforms(img) for img in examples["image"]]
    examples['labels'] = examples['label']
    del examples["image"]
    return examples

# self - training class
class self_training:
    def __init__(self):
        self.img_shape = 224
        self.labels = [0, 1]
        self.batch_size = 256
        self.lr = 1e-4
        self.num_epochs = 50
        self.device = 'cuda'

    def all_train_data_transfer(self, subjects, from_path, to_path, class_name):
        list_files = pd.Series(os.listdir(from_path))
        for name in subjects.dropna():
            try:
                res = list_files.str.contains(pat=name)
                relevant_files = list_files[res.values]
                for file in relevant_files:
                    shutil.copy(os.path.join(from_path, file), os.path.join(to_path, file))
            except:
                print(name, ' - Not exist!')
                continue
        print("All Train "+ class_name + " Transfer - Done!")

    def build_model(self, pretrained_model_path, mode):
        # model
        device = 'cuda'
        model_name_or_path = "microsoft/swin-base-patch4-window7-224-in22k"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        model = SwinForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(self.labels),
            id2label={str(i): c for i, c in enumerate(self.labels)},
            label2id={c: str(i) for i, c in enumerate(self.labels)}
        )
        if mode == "Predict":
            save_path = pretrained_model_path
            model.load_state_dict(torch.load(save_path))
        return model


    def data_preparation(self, dataset_path, mode):
        dataset = load_dataset("imagefolder", data_files={"train": str(dataset_path+'/**')},
                               drop_labels=False,
                               )
        if mode == "train":
            prepared_ds = dataset["train"].with_transform(train_data_transform)
        else:
            prepared_ds = dataset["train"].with_transform(data_transform)
        return prepared_ds

    def sig(self,x):
        return 1/(1 + np.exp(-x))

    def predict(self, model_path, data_path):
        device = 'cuda'
        model = self.build_model(model_path, "Predict")
        loader = self.data_preparation(data_path," ")
        probabilities = []
        true_labels = []
        model.eval()  # set to eval mode to avoid batchnorm
        with torch.no_grad():  # avoid calculating gradients
            correct, total = 0, 0
            for images in loader:
                labels = images['labels']
                img = images["pixel_values"]
                probabilities.append(self.sig(model(img[None,:]).logits.numpy()[0][1]))
                true_labels.append(labels)
        return probabilities, true_labels


    def ss_data_transfer(self, names, probs, class_name):
        from_folder = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/all_train"
        to_folder = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/semi-supervised_train"

        # transfer
        for file_name in names:
            from_path = os.path.join(from_folder, class_name, file_name)
            to_path = os.path.join(to_folder, class_name, file_name)
            shutil.copy(from_path, to_path)

        print(str(len(names)) + " files transferred to semi-supervised set in " + class_name + " class")
        return names, probs

    def train_model(self, train_path, val_path, save_name, load_path):
        print("start to train the model!")
        model = self.build_model(load_path, " ")
        dataset = self.data_preparation(train_path,"train")
        val_dataset = self.data_preparation(val_path," ")

        training_args = TrainingArguments(
            output_dir=save_name,
            per_device_train_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            num_train_epochs=self.num_epochs,
            fp16=True,
            save_strategy="epoch",
            learning_rate=self.lr,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to='tensorboard',
            load_best_model_at_end=True,
        )

        optimizer = AdamW(model.parameters(), lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=5000,num_training_steps=60000)
        optimizers = optimizer, scheduler

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            optimizers=optimizers,
        )
        #tokenizer=self.feature_extractor,
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

    def scan_based(self, file_names, probs, true):
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
        pred = np.where(ten_slice_prob < 0.5, 0, 1).reshape(len(ten_slice_prob), )
        return uniqe_filename, true_labels, pred, ten_slice_prob

    def case_based(self, file_names, probs, true):
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


    def calculate_matrix(self,true, pred, prob, mode):
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


    def evaluate(self, model_p, res_df, iteration):
        test_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Test"
        probabilities, true_labels = self.predict(model_p, test_path)
        # add to df
        res_df[str(iteration)] = probabilities

        # metrics
        # scan based
        test_files = os.listdir(os.path.join(test_path,"Negative")) + os.listdir(os.path.join(test_path,"Positive"))
        scan_names, scans_true, scans_pred, scans_prob = self.scan_based(test_files, probabilities, true_labels)
        self.calculate_matrix(scans_true, scans_pred, scans_prob, 'Scan-based Evaluation')

        # case base
        case_true, case_pred, case_prob = self.case_based(scan_names, scans_prob, scans_true)
        self.calculate_matrix(case_true, case_pred, case_prob, 'Case-based Evaluation')
        return res_df

    def val_pred(self, model_path, loader, loss_weight):
        total_loss = 0
        model = self.build_model(model_path, "Predict")
        probabilities = []
        true_labels = []
        model.eval()  # set to eval mode to avoid batchnorm
        with torch.no_grad():  # avoid calculating gradients
            correct, total = 0, 0
            for images in loader:
                labels = images['labels']
                true_labels.append(labels)
                labels = torch.tensor([labels])
                img = images["pixel_values"]
                outputs = model(img[None, :])
                probabilities.append(self.sig(outputs.logits.numpy()[0][1]))
                # compute custom loss (suppose one has 3 labels with different weights)
                loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, loss_weight]))
                loss = loss_fct(outputs.get("logits").view(-1, model.config.num_labels), labels.view(-1))
                total_loss += loss.item()

        probabilities = np.asarray(probabilities).reshape(len(probabilities), )
        true_labels = np.asarray(true_labels).reshape(len(true_labels), )
        predictions = np.where(probabilities < 0.5, 0, 1).reshape(len(probabilities), )
        return probabilities, predictions, true_labels, total_loss/len(true_labels)

    def val_eval(self, model_path, val_path, loss_weight):
        val_data = self.data_preparation(val_path, " ")

        probabilities, predictions, true_labels, loss = self.val_pred(model_path, val_data, loss_weight)
        print('Validation Loss - ', loss)
        self.calculate_matrix(true_labels, predictions, probabilities, "Validation evaluation")
        return loss







