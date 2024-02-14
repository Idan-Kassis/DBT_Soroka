import os
import shutil

data_path = '/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804'
unlabeled_path = '/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/unlabeled_train'

# Negative
neg_flag = False
if neg_flag:
    all_train = os.listdir(os.path.join(data_path, 'all_train/Negative'))
    labeled = os.listdir(os.path.join(data_path, 'labeled_data/Train/Negative'))

    unlabeled_train_neg = list(set(all_train) ^ set(labeled))

    for file in unlabeled_train_neg:
        shutil.copy(os.path.join(data_path, 'all_train/Negative', file), os.path.join(unlabeled_path, file))
    print('Unlabeled Negative - Transferred')


# Positive
all_train = os.listdir(os.path.join(data_path, 'all_train/Positive'))
labeled = os.listdir(os.path.join(data_path, 'labeled_data/Train/Positive'))

unlabeled_train_pos = list(set(all_train) ^ set(labeled))

for file in unlabeled_train_pos:
    shutil.copy(os.path.join(data_path, 'all_train/Positive', file), os.path.join(unlabeled_path, file))

print('Unlabeled Positive - Transferred')
