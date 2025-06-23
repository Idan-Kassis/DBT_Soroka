import torch
import numpy as np
from transformers import AutoFeatureExtractor, SwinForImageClassification
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from PIL import Image
import os
import pandas as pd


def predict_single_image(image_path, model, feature_extractor):
    # Define image preprocessing
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    #resize = Resize((feature_extractor.size['height'], feature_extractor.size['width']))
    resize = Resize((1024, 1024))
    transform = Compose([resize, ToTensor(), normalize])
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Define sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict
    with torch.no_grad():
        logits = model(image_tensor).logits.cpu().numpy()
        probability = sigmoid(logits[:, 1])
    return probability[0]


def predict_folder_images(folder_path, model_path, model_name_or_path="microsoft/swin-base-patch4-window12-384-in22k"):
    # Load the model and feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    model = SwinForImageClassification.from_pretrained(
        model_name_or_path,
        image_size=1024,
        ignore_mismatched_sizes=True,
        num_labels=2,  # Assuming 2 labels: 0 and 1
        id2label={"0": 0, "1": 1},
        label2id={0: "0", 1: "1"}
    )
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    predictions = []
    filenames = []

    # Iterate through files in folder
    for i, filename in enumerate(os.listdir(folder_path)):
        print(i)
        if filename.lower().endswith(('.png')):  # Filter image files
            image_path = os.path.join(folder_path, filename)
            prediction = predict_single_image(image_path, model, feature_extractor)
            print(prediction)
            predictions.append(prediction)
            filenames.append(filename)
            #print(f"Predicted probability for {filename}: {prediction:.4f}")
    return predictions, filenames


# Example usage
if __name__ == "__main__":
    folder_path = "/Users/idankassis/Desktop/Thesis/0_0-600"
    model_path = "/Users/idankassis/Desktop/Thesis/Swin-base-DBT-1024-4-lr4/pytorch_model.bin"
    probs, filenames = predict_folder_images(folder_path, model_path)
    true_list = [0] * len(filenames)
    results_train = pd.DataFrame({"Filename": filenames,
                                  "true_labels": np.array(true_list).reshape(len(true_list), ),
                                  "Probabilities": np.array(probs).reshape(len(probs), )})
    results_train.to_csv(str("negative_0-600_prediction_1024.csv"), index=False)
