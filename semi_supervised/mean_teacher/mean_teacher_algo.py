import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset
from transformers import SwinForImageClassification, AutoFeatureExtractor
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from datasets import load_dataset
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from transformers import AutoFeatureExtractor, SwinForImageClassification
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ToPILImage, RandomResizedCrop, RandomVerticalFlip, RandomRotation, RandomHorizontalFlip
from torch.utils.data import WeightedRandomSampler, RandomSampler
from itertools import zip_longest
from torchvision import datasets
import torchvision.transforms as T

gpu_number = 3
loss_name = "MSE"
#consistency_loss_fn = nn.CrossEntropyLoss()
consistency_loss_fn = nn.MSELoss()
ssl_weights = [10]
lr_array = [5,6]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
device = 'cuda'

# -------------------------------------  Step 1: Data Preparation --------------------------------------

# Step 1: Data Preparation
labeled_data_dir = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384"
unlabeled_data_dir = "/workspace/DBT_US_Soroka/semi-supervised_data/unlabeled_data-384"

# Define the transformation pipeline for data preprocessing and augmentation
color_jitter = T.ColorJitter(
            0.8 , 0.8 , 0.8 , 0.2
        )
# 10% of the image
blur = T.GaussianBlur((3, 3), (0.1, 2.0))

transform1 = Compose(
            [
            T.RandomResizedCrop(size=384),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.2),
            ]
        )

color_jitter = T.ColorJitter(
            0.8 , 0.8 , 0.8 , 0.2
        )
# 10% of the image
blur = T.GaussianBlur((3, 3), (0.1, 2.0))

transform2 = Compose(
            [
            T.RandomResizedCrop(size=384),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.2),
            ]
        )

model_name_or_path = "microsoft/swin-base-patch4-window12-384-in22k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transform = Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            RandomVerticalFlip(0.5),
                            RandomRotation(10),
                            RandomHorizontalFlip(0.5),
                            ToTensor(),
                            normalize])
val_transform = Compose([ToTensor(), normalize])

# unlabeled Dataset


class UnlabeledDataClass(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.list_of_paths = os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.list_of_paths)

    def __getitem__(self, x):
        image_path = os.path.join(self.path, self.list_of_paths[x])
        image = Image.open(image_path)
        image = self.transform(image)

        return image, 10


# Instantiate the labeled and unlabeled datasets
#labeled_dataset_train = datasets.ImageFolder(root=pseudo_data_dir, transform=val_transform)
labeled_dataset_train = datasets.ImageFolder(root=os.path.join(labeled_data_dir,'Train'), transform=val_transform)
print('Train data size - ', len(labeled_dataset_train))
labeled_dataset_val = datasets.ImageFolder(root=os.path.join(labeled_data_dir,'Val'), transform=val_transform)
print('Val data size - ', len(labeled_dataset_val))
unlabeled_dataset = UnlabeledDataClass(unlabeled_data_dir, val_transform)
print('Unlabeled data size - ', len(unlabeled_dataset))


print('Datasets created!')


# balancing every batch
class_freq = [len(os.listdir(os.path.join(labeled_data_dir, "Train/Negative"))), len(os.listdir(os.path.join(labeled_data_dir, "Train/Positive")))]
#class_freq = [len(os.listdir(os.path.join(pseudo_data_dir, "Negative"))), len(os.listdir(os.path.join(pseudo_data_dir, "Positive")))]
class_weights = [1.0/ freq for freq in class_freq]
weights = [class_weights[0]] * int(class_freq[0]) + [class_weights[1]] * int(class_freq[1])
samples_weights = torch.tensor(weights, dtype=torch.double)
balanced_sampler = WeightedRandomSampler(samples_weights, num_samples=len(labeled_dataset_train), replacement=True)
validation_sampler = RandomSampler(labeled_dataset_val)


# Create data loaders for labeled training, labeled validation, and unlabeled data
labeled_train_data_loader = DataLoader(
    labeled_dataset_train,
    batch_size=16,
    sampler=balanced_sampler,
    pin_memory=True
)
labeled_val_data_loader = DataLoader(
    labeled_dataset_val,
    batch_size=16,
    sampler=validation_sampler,
    pin_memory=True
)

unlabeled_data_loader = DataLoader(
    unlabeled_dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=True
)

# ------------------------------------ Step 2: Model Architecture ---------------------------------------------


class SwinTransformerStudent(nn.Module):
    def __init__(self):
        super(SwinTransformerStudent, self).__init__()
        # Load the feature extractor
        ft_flag = False
        save_path = "/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model-384/Swin-base-DBT-384-4-lr4/pytorch_model.bin"
        model_name_or_path = "microsoft/swin-base-patch4-window12-384-in22k"
        labels = [0, 1]
        #feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

        # Load the SwinForImageClassification model
        self.backbone = SwinForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
        #self.backbone.load_state_dict(torch.load(save_path))
        if ft_flag:
            num_layers = len(self.backbone.base_model.encoder.layers)
            print('Number of layers - ', num_layers)
            freeze_layers = num_layers - 1
            for name, param in self.backbone.named_parameters():
                if any(f"swin.encoder.layers.{freeze_layer}." in name for freeze_layer in range(freeze_layers)):
                    param.requires_grad = False
            for name, param in self.backbone.named_parameters():
                print(name, param.requires_grad)
            print('Weights Freeze!')

    def forward(self, inputs):
        out = self.backbone(inputs)
        logits = out.logits
        return logits


class SwinTransformerTeacher(nn.Module):
    def __init__(self):
        super(SwinTransformerTeacher, self).__init__()
        # Load the feature extractor
        save_path = "/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model-384/Swin-base-DBT-384-4-lr4/pytorch_model.bin"
        model_name_or_path = "microsoft/swin-base-patch4-window12-384-in22k"
        labels = [0, 1]
        #feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

        # Load the SwinForImageClassification model
        self.backbone = SwinForImageClassification.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
        #self.backbone.load_state_dict(torch.load(save_path))

    def forward(self, inputs):
        out = self.backbone(inputs)
        logits = out.logits
        return logits


# ------------------------------------- Step 3: Training Loop ----------------------------------------
rampup_flag = False

for ssl_weight in ssl_weights:
    origin_ssl_weight = ssl_weight
    for lr_num in lr_array:
        print('Self-supervised Weight - ', ssl_weight, '  ,  lr - ', lr_num)
        learning_rate = 1 * 10 ** (-lr_num)
        weight_decay = 0.001

        # Define the loss functions for labeled and unlabeled data
        labeled_loss_fn = nn.CrossEntropyLoss()


        # define the model
        student_model = SwinTransformerStudent()
        student_model = student_model.to(device)
        teacher_model = SwinTransformerTeacher()
        teacher_model = teacher_model.to(device)

        # Define the optimizer and learning rate schedule
        optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Training loop
        num_epochs = 50
        early_stopping_patience = 3
        best_val_loss = float('inf')
        print('Self-supervised Weight - ', ssl_weight,'  - lr - ', lr_num)

        #model saving setting
        new_directory = os.path.join(os.getcwd(),"MT-swin384-"+str(ssl_weight)+'-lr'+str(lr_num)+'-loss-'+loss_name)
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        best_model_path_s = os.path.join(new_directory,'student_model-ssl'+str(ssl_weight)+'-lr'+str(lr_num)+'-loss-'+loss_name+'.pth')
        best_model_path_t = os.path.join(new_directory,'teacher_model-ssl' + str(ssl_weight) + '-lr' + str(lr_num)+'-loss-'+loss_name+'.pth')

        print('Start Training on GPU ',str(gpu_number), ' !')
        print(len(labeled_train_data_loader))
        print(len(unlabeled_data_loader))

        for epoch in range(num_epochs):
            if rampup_flag:
                if epoch < 1:
                    ssl_weight = 0
                else:
                    ssl_weight = origin_ssl_weight
            flag = 0
            student_model.train()
            teacher_model.train()

            train_loss = 0
            total_train = 0
            correct_train = 0

            print(f"Epoch {epoch + 1}/{num_epochs}:")

            for batch_idx, ((labeled_images, labels), (unlabeled_images, _)) in enumerate(zip_longest(labeled_train_data_loader, unlabeled_data_loader, fillvalue=(None, None))):

                # Rest of your code
                if labeled_images is None:
                    break
                    flag = 1

                # create noise
                if flag == 0:
                    #labeled_images_1 = torch.stack([transform1(image) for image in labeled_images])
                    labeled_images_1 = torch.stack([image for image in labeled_images])
                    labeled_images_2 = torch.stack([transform2(image) for image in labeled_images])
                if unlabeled_images is not None:
                    #unlabeled_images_1 = torch.stack([transform1(image) for image in unlabeled_images])
                    unlabeled_images_1 = torch.stack([image for image in unlabeled_images])
                    unlabeled_images_2 = torch.stack([transform2(image) for image in unlabeled_images])

                # labeled loss calc
                if flag == 0:
                    # Load a batch of labeled data
                    labeled_images_1 = labeled_images_1.to(device)
                    labeled_images_2 = labeled_images_2.to(device)
                    labels = labels.to(device)
                    # forward
                    labeled_outputs = student_model(labeled_images_1)
                    # Calculate the supervised loss
                    labeled_loss = labeled_loss_fn(labeled_outputs, labels)

                    with torch.no_grad():
                        teacher_outputs = teacher_model(labeled_images_2)
                    labeled_consistency_loss = consistency_loss_fn(torch.sigmoid(teacher_outputs), torch.sigmoid(labeled_outputs))

                else:
                    labeled_loss = 0
                    labeled_consistency_loss = 0


                # consistency loss calc
                # Load a batch of unlabeled data
                if unlabeled_images is not None:
                    unlabeled_images_1 = unlabeled_images_1.to(device)
                    unlabeled_images_2 = unlabeled_images_2.to(device)

                    # Forward pass (student model)
                    unlabeled_outputs = student_model(unlabeled_images_1)

                    # Forward pass (teacher model)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(unlabeled_images_2)

                    # Loss
                    # Calculate the consistency loss
                    consistency_loss = consistency_loss_fn(torch.sigmoid(teacher_outputs), torch.sigmoid(unlabeled_outputs))
                else:
                    consistency_loss=0
                # Calculate the total loss
                total_loss = labeled_loss + (ssl_weight * (torch.abs(consistency_loss+labeled_consistency_loss)))

                # Backpropagation and weight update
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()


                # Update the teacher model weights
                update_rate = 0.991  # Exponential moving average update rate
                for param_student, param_teacher in zip(student_model.parameters(), teacher_model.parameters()):
                    param_teacher.data = update_rate * param_teacher.data + (1 - update_rate) * param_student.data

                train_loss += total_loss.item()

                if flag == 0:
                    _, predicted_train = torch.max(labeled_outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted_train == labels).sum().item()
            train_loss /= len(labeled_train_data_loader)
            train_accuracy = 100 * correct_train / total_train

            # Validation
            student_model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0

            with torch.no_grad():

                #print('Start Validation')

                for val_images, val_labels in labeled_val_data_loader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = student_model(val_images)
                    labeled_loss = labeled_loss_fn(val_outputs, val_labels)

                    loss = labeled_loss
                    val_loss += loss.item()

                    _, predicted_val = torch.max(val_outputs.data, 1)
                    total_val += val_labels.size(0)
                    correct_val += (predicted_val == val_labels).sum().item()

            val_loss /= len(labeled_val_data_loader)
            val_accuracy = 100 * correct_val / total_val

            # Save the best model if a new best validation loss is achieved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(student_model.state_dict(), best_model_path_s)
                torch.save(teacher_model.state_dict(), best_model_path_t)
                early_stopping_counter = 0  # Reset the counter
            else:
                early_stopping_counter += 1

            # Print the training and validation process
            #print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
            print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            # Check for early stopping
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered. Training stopped.")
                break

            # Learning rate schedule update
            lr_scheduler.step()


# ------------------------------------------- Step 4: Evaluation --------------------------------------------------
pred_flag = False
if pred_flag:
    student_model.load_state_dict(torch.load(best_model_path))
    student_model.eval()
    test_data_dir = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384/Test"
    test_transform = Compose([ToTensor(), normalize])
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)
    test_data_loader = DataLoader(test_dataset, batch_size=1)

    with torch.no_grad():
        correct = 0
        total = 0
        positive_probs = []

        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = student_model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Calculate probabilities using softmax function
            softmax_probs = torch.softmax(outputs, dim=1)
            positive_probs.extend(softmax_probs[:, 1].tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy}%")


    # Convert the positive_probs list to a numpy array
    positive_probs = np.array(positive_probs)
