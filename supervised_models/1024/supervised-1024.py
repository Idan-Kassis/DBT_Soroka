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

#  -------------------------------------------- Parameters ------------------------------------------------------------
gpu_number = 3
dataset_path = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-1024"
labels = [0, 1]
#model_types = ['Swin-base', 'Swin-large'] # bs=64
model_types = ['Swin-base']
batch_size = 8
patch_size = 4
lr_array = [4]

for model_type in model_types:
    for cur_lr in lr_array:
        lr_str = str(cur_lr)
        lr = 1 * 10 ** (-cur_lr)
        save_name = str(model_type+"-DBT-384-"+str(patch_size)+"-lr"+lr_str)

        clsss_proportion = len(os.listdir(os.path.join(dataset_path,"Train/Negative")))/len(os.listdir(os.path.join(dataset_path,"Train/Positive")))
        # ------------------------------------------- Define the Data ---------------------------------------------------------

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
        dataset = load_dataset("imagefolder", data_files={"train": os.path.join(dataset_path,"Train/**"),
                                                          "test": os.path.join(dataset_path,"Test/**"),
                                                          "valid": os.path.join(dataset_path,"Val/**")},
                                drop_labels=False,
        )

        # ------------------------------------------- Define the Model --------------------------------------------------------
        if model_type == 'Swin-base':
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


        # ------------------------------------------ Data Functions ---------------------------------------------------
        normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        #normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        _train_transform = Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                    RandomVerticalFlip(0.5),
                                    RandomRotation(10),
                                    RandomHorizontalFlip(0.5),
                                    ToTensor(),
                                    normalize])
        _transforms = Compose([ToTensor(), normalize])

        def train_transform(examples):
            examples["pixel_values"] = [_train_transform(img) for img in examples["image"]]
            examples['labels'] = examples['label']
            del examples["image"]
            return examples

        def transform(examples):
            examples["pixel_values"] = [_transforms(img) for img in examples["image"]]
            examples['labels'] = examples['label']
            del examples["image"]
            return examples

        def collate_fn(batch):
            return {
                'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
                'labels': torch.tensor([x['labels'] for x in batch]),
            }

        # data preparation
        prepared_ds = dataset["train"].with_transform(train_transform)
        prepared_ds_val = dataset["valid"].with_transform(transform)

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


        # ------------------------------------------ Model Training -------------------------------------------------------
        training_args = TrainingArguments(
          output_dir=save_name,
          per_device_train_batch_size=batch_size,
          evaluation_strategy="epoch",
          logging_strategy="epoch",
          num_train_epochs=50,
          fp16=True,
          save_strategy="epoch",
          learning_rate=lr,
          save_total_limit=2,
          remove_unused_columns=False,
          push_to_hub=False,
          report_to='tensorboard',
          load_best_model_at_end=True,
        )

        optimizer=AdamW(model.parameters(), lr=lr)
        scheduler=transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=1000,num_training_steps=60000)
        optimizers = optimizer, scheduler

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, clsss_proportion]).to("cuda"))
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=prepared_ds,
            eval_dataset=prepared_ds_val,
            tokenizer=feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            optimizers=optimizers,
        )

        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        # ------------------------------------ Model evaluation on validation  -----------------------------------------------
        print('--------------- Model - ', model_type)
        print("-------------- Current Learning Rate - ", str(cur_lr))
        metrics = trainer.evaluate(prepared_ds_val)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
