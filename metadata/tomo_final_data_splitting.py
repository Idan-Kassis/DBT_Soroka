import os
import pandas as pd

pos_subjects = pd.read_csv('/workspace/DBT_US_Soroka/Manually_Selected_Frames/all_tomo_pos_scans.csv')
pos_subjects = pos_subjects['study_ID']
uniqe_pos = []
for name in pos_subjects:
    if name[:6] not in uniqe_pos:
        uniqe_pos.append(name[:6])

neg_list = os.listdir("/workspace/DBT_US_Soroka/Preprocessed_2D/All_Segmented/Negative")
uniqe_neg = []
for name in neg_list:
    if name[:6] not in uniqe_neg:
        uniqe_neg.append(name[:6])



import random
random.shuffle(uniqe_pos)
train_pos = uniqe_pos[:round(0.7*len(uniqe_pos))]
val_pos = uniqe_pos[round(0.7*len(uniqe_pos)):round(0.85*len(uniqe_pos))]
test_pos = uniqe_pos[round(0.85*len(uniqe_pos)):]

train_neg = []
val_neg = []
test_neg = []
num_train_neg = round(0.7*len(uniqe_neg))
num_val_neg = round(0.15*len(uniqe_neg))
random.shuffle(uniqe_neg)
to_check = []
for name in uniqe_neg:
    if name in train_pos:
        train_neg.append(name)
    elif name in val_pos:
        val_neg.append(name)
    elif name in test_pos:
        test_neg.append(name)
    else:
        to_check.append(name)

for name in to_check:
    if len(train_neg) < num_train_neg:
        train_neg.append(name)
    elif len(val_neg) < num_val_neg:
        val_neg.append(name)
    else:
        test_neg.append(name)

df = pd.concat([pd.Series(train_pos),pd.Series(train_neg), pd.Series(val_pos), pd.Series(val_neg), pd.Series(test_pos), pd.Series(test_neg)], ignore_index=True, axis=1)
df.columns =['train_positive', 'train_negative', 'val_positive', 'val_negative', 'test_positive', 'test_negative']
df.to_excel('tomo_subjects_sets_all_2804.xlsx',index = False)

