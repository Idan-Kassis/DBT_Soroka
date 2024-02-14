import os
import shutil
from datasets import load_dataset
import random
from PIL import ImageDraw, ImageFont, Image
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
from torch.utils.data import DataLoader

# training functions
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch]),
    }


def compute_metrics(p):
    metric1 = load_metric("accuracy")
    metric2 = load_metric("precision")
    metric3 = load_metric("recall")
    metric4 = evaluate.load("roc_auc")

    accuracy = metric1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["accuracy"]
    precision = metric2.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["precision"]
    recall = metric3.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["recall"]
    auc = metric4.compute(references=p.label_ids, prediction_scores=np.argmax(p.predictions, axis=1))["roc_auc"]
    return {"accuracy": accuracy, "PPV": precision, "sensitivity": recall, "roc_auc": auc}


class CustomTrainer(Trainer):
    def calculate_loss_weights(self, train_path="/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/semi-supervised_train"):
        num_pos = len(os.listdir(os.path.join(train_path,'Positive')))
        num_neg = len(os.listdir(os.path.join(train_path,'Negative')))
        return num_neg/num_pos

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, self.calculate_loss_weights()]).to('cuda'))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
