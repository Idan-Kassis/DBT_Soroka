
from transformers import SwinForImageClassification
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
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
# YM7414_L_MLO_40
# AN5736_R_CC_43
# AS7237_R_CC_20
# AS7815_R_MLO_20
# SZ6296_R_CC_19

# Test/Positive/SG6876_L_CC_12.png


image_path = '/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384/Test/Negative/GN2071_R_MLO_61.png'
original_image_path = '/workspace/DBT_US_Soroka/Preprocessed_2D/All_Segmented/Negative/GN2071_R_MLO_61.png'
original_image = np.float32(cv2.imread(original_image_path)) / 255
image = cv2.imread(image_path)
#image = cv2.resize(image, (224, 224))
image = np.float32(image) / 255
#img_tensor = preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_tensor = transforms.ToTensor()(image)
img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)


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
                          method: Callable = EigenCAM):
    with method(model=HuggingfaceToTensorModelWrapper(model),
                target_layers=[target_layer],
                reshape_transform=reshape_transform) as cam:
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)

        grayscale_cam = batch_results[0,:]
        visualization = show_cam_on_image(np.float32(input_image), grayscale_cam, use_rgb=True, image_weight=0.8)
        imgplot = plt.imshow(visualization)
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('gradcam.jpg', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        plt.clf()

        interpolated = F.interpolate(torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0), size=(original_image.shape[0],original_image.shape[1]), mode="bilinear", align_corners=False)
        visualization = show_cam_on_image(np.float32(original_image), interpolated.squeeze(0).squeeze(0), use_rgb=True, image_weight=0.85)
        imgplot = plt.imshow(visualization)
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('gradcam_interpolated.jpg', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        plt.clf()

        '''
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image) / 255,
                                              grayscale_cam,
                                              use_rgb=True)
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1] // 2, visualization.shape[0] // 2))
            results.append(visualization)
        return np.hstack(results)
        '''

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


def swinT_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

''''''
labels = [0,1]
model_name_or_path = "microsoft/swin-base-patch4-window12-384-in22k"
model = SwinForImageClassification.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
save_path = "/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model-384/Swin-base-DBT-384-4-lr4/pytorch_model.bin"
#save_path = "/workspace/DBT_US_Soroka/Codes/DBT/2D/self_supervised_mean-teacher/ssl-weight-384-experiment/pseudo_exp/MT-swin384-1-lr8-loss-CE-PSEUDO-EXT/student_model-ssl1-lr8-loss-CE-PSEUDO-EXT.pth"

model.load_state_dict(torch.load(save_path))
#target_layer = model.swin.encoder.layers[-1].blocks[-1].layernorm_after
target_layer = model.swin.layernorm

targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, 1))]
reshape_transform = partial(swinT_reshape_transform_huggingface,
                            width=img_tensor.shape[2]//32,
                            height=img_tensor.shape[1]//32)
'''
display(Image.fromarray(run_dff_on_image(model=model,
                          target_layer=target_layer,
                          classifier=model.classifier,
                          img_pil=image,
                          img_tensor=img_tensor,
                          reshape_transform=reshape_transform,
                          n_components=4,
                          top_k=1)))
display(Image.fromarray(run_grad_cam_on_image(model=model,
                      target_layer=target_layer,
                      targets_for_gradcam=targets_for_gradcam,
                      reshape_transform=reshape_transform)))
'''

run_grad_cam_on_image(model=model,
                      target_layer=target_layer,
                      targets_for_gradcam=targets_for_gradcam,
                      reshape_transform=reshape_transform)

'''
cam = run_grad_cam_on_image(model=model,
                      target_layer=target_layer,
                      targets_for_gradcam=targets_for_gradcam,
                      reshape_transform=reshape_transform)

#print_top_categories(model, img_tensor)

#cam_image = show_cam_on_image(rgb_img, cam)
cv2.imwrite('swin_grad_cam.jpg', cam)
'''