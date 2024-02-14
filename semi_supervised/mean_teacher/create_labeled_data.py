import os
import shutil
import pandas as pd
import random
from PIL import Image
import numpy as np
import os
sets_df = pd.read_excel("/workspace/DBT_US_Soroka/Codes/DBT/2D/tomo_subjects_sets_all_2804.xlsx")
experiment_path = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-1024"
pos_flag = True
neg_flag = True
resize_flag = True
# Pos
if pos_flag:
    # train
    subjects = sets_df["train_positive"]
    from_path = "/workspace/DBT_US_Soroka/Manually_Selected_Frames/Positive"
    to_path = os.path.join(experiment_path, "Train/Positive")

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
    print("Train Positive - Done!")


    # val
    subjects = sets_df["val_positive"]
    from_path = "/workspace/DBT_US_Soroka/Manually_Selected_Frames/Positive"
    to_path = os.path.join(experiment_path, "Val/Positive")

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
    print("Val Positive - Done!")

    # test
    subjects = sets_df["test_positive"]
    from_path = "/workspace/DBT_US_Soroka/Preprocessed_2D/All_Segmented/Positive"
    to_path = os.path.join(experiment_path, "Test/Positive")
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
    print("Test Positive - Done!")

# Neg
if neg_flag:
    from_path = "/workspace/DBT_US_Soroka/Preprocessed_2D/All_Segmented/Negative"

    # train
    subjects = sets_df["train_negative"]
    to_path = os.path.join(experiment_path, "Train/Negative")
    list_files = pd.Series(os.listdir(from_path))
    for name in subjects.dropna():
        try:
            res = list_files.str.contains(pat=name)
            relevant_files = list_files[res.values]
            type_scans = []
            for file in relevant_files:
                if file[:12] not in type_scans:
                    type_scans.append(file[:12])
            for scan in type_scans:
                res = list_files.str.contains(pat=scan)
                relevant_files = list_files[res.values]
                relevant_files = relevant_files.tolist()
                random.shuffle(relevant_files)
                for file in relevant_files[:10]:
                    shutil.copy(os.path.join(from_path, file), os.path.join(to_path, file))
        except:
            print(name, ' - Not exist!')
            continue
    print("Train Negative - Done!")

    # val
    subjects = sets_df["val_negative"]
    to_path = os.path.join(experiment_path, "Val/Negative")
    list_files = pd.Series(os.listdir(from_path))
    for name in subjects.dropna():
        try:
            res = list_files.str.contains(pat=name)
            relevant_files = list_files[res.values]
            type_scans = []
            for file in relevant_files:
                if file[:12] not in type_scans:
                    type_scans.append(file[:12])
            for scan in type_scans:
                res = list_files.str.contains(pat=scan)
                relevant_files = list_files[res.values]
                relevant_files = relevant_files.tolist()
                random.shuffle(relevant_files)
                for file in relevant_files[:10]:
                    shutil.copy(os.path.join(from_path, file), os.path.join(to_path, file))
        except:
            print(name, ' - Not exist!')
            continue
    print("Val Negative - Done!")

    # test

    subjects = sets_df["test_negative"]
    to_path = os.path.join(experiment_path, "Test/Negative")
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
    print("Test Negative - Done!")




if resize_flag:
    sets = ["Train", "Val", "Test"]
    classes = ["Negative", "Positive"]

    for cur_set in sets:
        for label in classes:
            files = os.listdir(os.path.join(experiment_path, cur_set, label))
            for f in files:
                img = Image.open(os.path.join(experiment_path, cur_set, label, f))

                # Resize the image to 224x224 pixels
                img = img.resize((1024, 1024), resample=Image.BILINEAR)

                # Convert the image to a NumPy array
                img_arr = np.array(img)

                # Save the normalized image as a PNG file
                Image.fromarray(np.uint8(img_arr)).save(os.path.join(experiment_path, cur_set, label, f))

            print("Done - ", ' - ', cur_set, ' - ', label)
