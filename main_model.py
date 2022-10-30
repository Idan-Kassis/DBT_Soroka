# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:54:36 2022

@author: Dror
"""
# Setup

from data_generator_tomo import balanced_data_generator, data_generator
from model_DBT import model_tomo_3D
import pandas as pd
from sklearn.model_selection import train_test_split
from clearml import Task
import tensorflow as tf

# set gpu and clearML
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus, 'GPU')
Task.init(project_name='DBT-SOROKA', task_name='Training_2710')

# Define parameters
batch_size = 8
num_classes = 2
learning_rate = 0.0001
epochs = 500
input_size = [25,200,200]
train_pr = 0.6
val_pr = 0.2
test_pr = 0.2
# -----------------------------------------------------------------------------
# Read Excel

df = pd.read_excel('Soroka_DBT_Metadata_Dicom.xlsx')


# Remove Invalid data
#invalid_list = ["/mnt/md0/databases/DBT_US_Soroka/Anonymouse Data/KY2807/TOMO/WSCSFUVQ/52YQVNYJ",
#                "/mnt/md0/databases/DBT_US_Soroka/Anonymouse Data/AA5258/TOMO/AHFQTQZB/0NVO053E",
#                "/mnt/md0/databases/DBT_US_Soroka/Anonymouse Data/RI0924/TOMO/V3MCX5J4/15THUGIA",
#                "/mnt/md0/databases/DBT_US_Soroka/Anonymouse Data/RY1163/TOMO/EVTUTFMB/LM3UH54S"]
invalid_list = ["/data/Breast_Cancer/DBT_US_Soroka/Anonymouse Data/KY2807/TOMO/WSCSFUVQ/52YQVNYJ",
                "/data/Breast_Cancer/DBT_US_Soroka/Anonymouse Data/AA5258/TOMO/AHFQTQZB/0NVO053E",
                "/data/Breast_Cancer/DBT_US_Soroka/Anonymouse Data/RI0924/TOMO/V3MCX5J4/15THUGIA",
                "/data/Breast_Cancer/DBT_US_Soroka/Anonymouse Data/RY1163/TOMO/EVTUTFMB/LM3UH54S"]
for invalid in invalid_list:
    invalid_loc = df.loc[df['Images Path']==invalid].index
    df = df.drop(invalid_loc[0],axis=0)

    
# split dataframe to subsets
train_df, rest = train_test_split(df, test_size=0.4)
val_df, test_df = train_test_split(rest, test_size=0.5)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Create data generator
train_gen = balanced_data_generator(train_df, batch_size, input_size)
val_gen = data_generator(val_df, batch_size, input_size)
test_gen = data_generator(test_df, batch_size, input_size)

# get the model
model_tomo = model_tomo_3D(train_gen, val_gen, test_gen, learning_rate, epochs, input_size)
model_tomo.build_model()

# train the model
model_tomo.train_model()




# #%%
# # test data
# test_data = np.zeros([len(list_test),30,180,180])
# for i in range(len(list_test)):
#     test_data[i,:,:,:] = np.load(os.path.join(os.getcwd(),'test',classes_name[list_test_labels[i]],list_test[i]))


# # prediction
# y_pred = model_tomo.prediction(test_data)
# y_classes = np.argmax(y_pred, axis=1)

# y_true = test_gen.y_test


# from classification_models_3D.tfkeras import Classifiers
# Network, preprocess_input = Classifiers.get('vgg19')
# model = Network(input_shape=(40, 160, 160, 1), 
#                       weights='imagenet')


# Network, preprocess_input = Classifiers.get('resnet18')
# model = Network(
#    input_shape=(30, 224, 224,1),
#    stride_size=4,
#    kernel_size=3, 
#    weights='imagenet'
# )


# from more_itertools import locate

# def find_indices(list_to_check, item_to_find):
#     indices = locate(list_to_check, lambda x: x == item_to_find)
#     print(indices)
#     return list(indices)
 
# path_list = df['Images Path']
# indexes_positive = find_indices(df['Mass'], 'P')
# list_path_neg = path_list[indexes_positive]

# ind_neg = [0, 1,2]
# list_path_neg[2] for k in ind_neg]

