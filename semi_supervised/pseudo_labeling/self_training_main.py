import os
import pandas as pd
import numpy as np
from self_training_functions import self_training
from datasets import load_dataset
import random
from PIL import ImageDraw, ImageFont, Image
import torch
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
import shutil
import warnings
# Ignore all warning messages
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
all_train_flag = False
add_labeled = False
resize_flag = False
st = self_training()

if add_labeled:
    categories = ["Negative", "Positive"]
    from_p = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Train"
    to_p = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/semi-supervised_train"
    for cat in categories:
        files = os.listdir(os.path.join(from_p, cat))
        for file in files:
            shutil.copy(os.path.join(from_p, cat, file), os.path.join(to_p, cat, file))
    print("labeled data transferred - Done!")


if all_train_flag:
    experiment_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/all_train"
    sets_df = pd.read_excel("/workspace/DBT_US_Soroka/Codes/DBT/2D/tomo_subjects_sets_all_2804.xlsx")
    # Positive
    subjects_pos = sets_df["train_positive"]
    from_path = "/workspace/DBT_US_Soroka/Preprocessed_2D/All_Segmented/Positive"
    to_path = os.path.join(experiment_path, "Positive")
    st.all_train_data_transfer(subjects_pos, from_path, to_path, "Positive")
    # Negative
    subjects_pos = sets_df["train_negative"]
    from_path = "/workspace/DBT_US_Soroka/Preprocessed_2D/All_Segmented/Negative"
    to_path = os.path.join(experiment_path, "Negative")
    st.all_train_data_transfer(subjects_pos, from_path, to_path, "Negative")

if resize_flag:
    classes = ["Negative", "Positive"]
    experiment_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/all_train"
    for label in classes:
        files = os.listdir(os.path.join(experiment_path, label))
        for f in files:
            img = Image.open(os.path.join(experiment_path, label, f))

            # Resize the image to 224x224 pixels
            img = img.resize((224, 224), resample=Image.BILINEAR)

            # Convert the image to a NumPy array
            img_arr = np.array(img)

            # Save the normalized image as a PNG file
            Image.fromarray(np.uint8(img_arr)).save(os.path.join(experiment_path, label, f))

        print("Done Resize - " ,label)

# ----------------------------------------------------------------------------------------------------------------------
model_path_list = ['/workspace/DBT_US_Soroka/Codes/DBT/2D/supervised_model/Swin-DBT-224-4-lr4/pytorch_model.bin']
all_data_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/all_train"
experiment_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/all_train"
neg_files = os.listdir(os.path.join(experiment_path, "Negative"))
pos_files = os.listdir(os.path.join(experiment_path, "Positive"))
results_df = pd.DataFrame()
val_loss_array = []

for iteration in range(0,10):
    print ("=========================   Iteration Number " + str(iteration) + "   =========================")
    # prediction on supervised model
    # Train
    model_path = model_path_list[iteration]
    probs, true = st.predict(model_path, all_data_path)
    df = pd.DataFrame({"Filename": neg_files+pos_files,
                            "true_labels": np.array(true).reshape(len(true), ),
                            "Probabilities": np.array(probs).reshape(len(probs), )})
    csv_name = 'iteration_' + str(iteration) + '_train_pred.csv'
    df.to_csv(csv_name, index=False)
    print("All train prediction - Done!")

    # slices to add
    # remove from df the exist in semi-supervised train set
    ss_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/semi-supervised_train"
    exist_ss_train = os.listdir(os.path.join(ss_path,"Negative")) + os.listdir(os.path.join(ss_path,"Positive"))
    for f in exist_ss_train:
        df.drop(df.loc[df['Filename'] == f].index, inplace=True)

    arr = df[df["true_labels"] == 0]["Probabilities"].to_numpy()
    names_add_neg = list(df[np.logical_and(df["true_labels"] == 0, df["Probabilities"] < np.percentile(arr, 2))]["Filename"].values)
    probs_neg = list(df[np.logical_and(df["true_labels"] == 0, df["Probabilities"] < np.percentile(arr, 2))]["Probabilities"].values)
    arr = df[df["true_labels"] == 1]["Probabilities"].to_numpy()
    names_add_pos = list(df[np.logical_and(df["true_labels"] == 1, df["Probabilities"] > np.percentile(arr, 99))]["Filename"].values)
    probs_pos = list(df[np.logical_and(df["true_labels"] == 1, df["Probabilities"] > np.percentile(arr, 99))]["Probabilities"].values)

    # transfer files to semi-supervised train (if not in the folder yet)
    files_neg, probs_neg = st.ss_data_transfer(names_add_neg, probs_neg, 'Negative')
    files_pos, probs_pos = st.ss_data_transfer(names_add_pos, probs_pos, 'Positive')
    true = [0]*len(files_neg) + [1]*len(files_pos)
    probs = probs_neg + probs_pos
    # save
    csv_name = 'iteration_'+str(iteration)+'_added_files.csv'
    add_df = pd.DataFrame({"Filename": files_neg + files_pos,
                            "true_labels": np.array(true).reshape(len(true), ),
                            "Probabilities": np.array(probs).reshape(len(probs), )})
    add_df.to_csv(csv_name, index=False)


    # train on semi-supervised data
    # create new model name , save model + add to list
    train_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/semi-supervised_train"
    val_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data/Val"
    current_model_name = str("./iteration_"+str(iteration)+"_model")
    model_path_list.append(str('/workspace/DBT_US_Soroka/Codes/DBT/2D/semi-supervised/self_training_swin/'+current_model_name+'/pytorch_model.bin'))
    st.train_model(train_path, val_path, current_model_name, model_path_list[iteration])

    # validation
    weight_loss = len(os.listdir(os.path.join(train_path, 'Negative')))/len(os.listdir(os.path.join(train_path, 'Positive')))
    vsl_loss = st.val_eval(model_path_list[iteration+1], val_path, weight_loss)
    val_loss_array.append(vsl_loss)

# evaluate - on all test and save probabilities in df, calculate res and display
min_index = val_loss_array.index(min(val_loss_array))
results_df = st.evaluate(model_path_list[min_index+1],results_df, min_index)
#results_df.to_csv("semi-supervised_test_prediction.csv", index=False)
print("Evaluation - Done!")




