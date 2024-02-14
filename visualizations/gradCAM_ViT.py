
from transformers import SwinForImageClassification, ViTForImageClassification
from functools import partial
import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from datasets import load_dataset
from pytorch_grad_cam import run_dff_on_image, GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

image_path = '/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Test/Positive/KE3811_L_CC_20.png'
image = cv2.imread(image_path)
#image = cv2.resize(image, (224, 224))
image = np.float32(image) / 255
img_tensor = transforms.ToTensor()(image)


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]


def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module = img_tensor,
                          input_image: Image = image,
                          method: Callable = GradCAM):
    with method(model=HuggingfaceToTensorModelWrapper(model),
                target_layers=[target_layer],
                reshape_transform=reshape_transform) as cam:
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        grayscale_cam = batch_results[0,:]
        visualization = show_cam_on_image(np.float32(input_image), grayscale_cam)
        imgplot = plt.imshow(visualization)
        plt.savefig('gradcam_VIT.jpg')
        plt.close()
        plt.clf()


def print_top_categories(model, img_tensor, top_k=2):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k:][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        return torch.sigmoid(model_output[int(self.category)])


def reshape_transform_vit_huggingface(x):
    activations = x[:, 1:, :]
    activations = activations.view(activations.shape[0],
                                   14, 14, activations.shape[2])
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations

labels = [0,1]
model_name_or_path = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
save_path = "/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model/ViT-DBT-224-16-lr5/pytorch_model.bin"
model.load_state_dict(torch.load(save_path))
target_layer_gradcam = model.vit.encoder.layer[-2].output


targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, 1))]


run_grad_cam_on_image(model=model,
                      target_layer=target_layer_gradcam,
                      targets_for_gradcam=targets_for_gradcam,
                      input_tensor=img_tensor,
                      input_image=image,
                      reshape_transform=reshape_transform_vit_huggingface)

'''
cam = run_grad_cam_on_image(model=model,
                      target_layer=target_layer,
                      targets_for_gradcam=targets_for_gradcam,
                      reshape_transform=reshape_transform)

#print_top_categories(model, img_tensor)

#cam_image = show_cam_on_image(rgb_img, cam)
cv2.imwrite('swin_grad_cam.jpg', cam)
'''