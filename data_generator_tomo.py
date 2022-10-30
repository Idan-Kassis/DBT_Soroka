# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:24:18 2022

@author: Idan
"""

import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import random
import pydicom
from more_itertools import locate
from preprocessing_DBT import preprocess_3D
import pandas as pd

# ----------------------------- Train ----------------------------------------
class balanced_data_generator(tf.compat.v2.keras.utils.Sequence):
  def __init__(self, df ,batch_size, dim, n_classes=2):
    # Constructor of the data generator.
    self.path_list = df['Images Path']
    self.class_list = df['Mass']
    self.dim = dim
    self.batch_size = batch_size
    self.half_batch_size = int(batch_size/2)
    self.n_classes = n_classes
    self.indexes_negative = self.find_indices(self.class_list, 'N')
    self.indexes_positive = self.find_indices(self.class_list, 'P')
    self.list_path_neg =     pd.Series.tolist(self.path_list[self.indexes_negative])
    self.list_path_pos = pd.Series.tolist(self.path_list[self.indexes_positive])
    self.on_epoch_end()

    
  def find_indices(self, list_to_check, item_to_find):
      indices = locate(list_to_check, lambda x: x == item_to_find)
      return list(indices)
 
  def __len__(self):
    # Denotes the number of batches per epoch
    return int(np.floor(len(self.list_path_neg) / self.half_batch_size))

  def __getitem__(self, index):
    # Generate one batch of data
    ind_neg = self.ind_neg[index*self.half_batch_size:(index+1)*self.half_batch_size]
    if len(self.list_path_pos)-4 < self.ind_neg[index*self.half_batch_size]:
        start = random.randint(0,len(self.list_path_pos)-4)
        ind_pos = self.ind_pos[start : start+self.half_batch_size]
    else: 
        ind_pos = self.ind_pos[index*self.half_batch_size:(index+1)*self.half_batch_size]
    # Find list of IDs
    list_neg_temp = [self.list_path_neg[k] for k in ind_neg]
    list_pos_temp = [self.list_path_pos[j] for j in ind_pos]
    list_path_temp = list_neg_temp + list_pos_temp
    list_labels_temp = [0] * self.half_batch_size + [1] * self.half_batch_size
    
    # shuffle
    temp = list(zip(list_path_temp, list_labels_temp))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    list_path_temp, list_labels_temp = list(res1), list(res2)
    # Generate data
    X, y = self.__data_generation(list_path_temp, list_labels_temp)

    return X, y

  def on_epoch_end(self):
    # This function is called at the end of each epoch.
    self.ind_neg = np.arange(len(self.list_path_neg))
    self.ind_pos = np.arange(len(self.list_path_pos))
    #if self.shuffle == True:
    #  np.random.shuffle(self.indexes)

  def __data_generation(self, list_path_temp, list_labels_temp):
    # Load individual numpy arrays and aggregate them to a batch.
    # For 3D
    X = np.empty([self.batch_size, self.dim[0], self.dim[1], self.dim[2]], dtype=np.float32)
    
    # y is a one-hot encoded vector.
    y = np.empty([self.batch_size, self.n_classes], dtype=np.int16)
    for i, cur_path in enumerate(list_path_temp):
        list_temp = os.listdir(cur_path)
        list_temp.remove("VERSION")
        dcm_list = [os.path.join(cur_path,f) for f in list_temp]
        # Load sample
        first = pydicom.dcmread(dcm_list[0]).pixel_array
        sahpe = first.shape
        org_dbt = np.zeros([sahpe[0], sahpe[1], len(dcm_list)])
        for l in range(len(dcm_list)):
            org_dbt[:,:,l] = pydicom.dcmread(dcm_list[l]).pixel_array
        # preprocessing
        preprocess_dbt = preprocess_3D(org_dbt, self.dim[0], self.dim[1], self.dim[2])
        X[i,:, :, :] = preprocess_dbt.preprocessing()
        # Load labels       
        y[i,1] = list_labels_temp[i]
        y[i,0] = 1-list_labels_temp[i]
        
    return X, y

#%%
# ------------------------------ Rest --------------------------------------
class data_generator(tf.compat.v2.keras.utils.Sequence):
  def __init__(self, df ,batch_size, dim, n_classes=2):
    # Constructor of the data generator.
    self.dim = dim
    self.batch_size = batch_size
    self.list_examples = pd.Series.tolist(df['Images Path'])
    self.str_labels = pd.Series.tolist(df['Mass'])
    self.list_labels = np.zeros(len(self.str_labels))
    self.list_labels[self.find_indices(self.str_labels,'P')] == 1
    self.n_classes = n_classes
    self.on_epoch_end()

  def find_indices(self, list_to_check, item_to_find):
      indices = locate(list_to_check, lambda x: x == item_to_find)
      return list(indices)
  
  def __len__(self):
    # Denotes the number of batches per epoc
    return int(np.floor(len(self.list_examples) / self.batch_size))

  def __getitem__(self, index):
    # Generate one batch of data
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_examples[k] for k in indexes]
    list_labels_temp = [self.list_labels[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp, list_labels_temp)

    return X, y

  def on_epoch_end(self):
    # This function is called at the end of each epoch.
    self.indexes = np.arange(len(self.list_examples))
    

  def __data_generation(self, list_IDs_temp, list_labels_temp):
    # Load individual numpy arrays and aggregate them to a batch.
    
    X = np.empty([self.batch_size, self.dim[0], self.dim[1], self.dim[2]], dtype=np.float32)
    
    # y is a one-hot encoded vector.
    y = np.empty([self.batch_size, self.n_classes], dtype=np.int16)
    for i, cur_path in enumerate(list_IDs_temp):
        list_temp = os.listdir(cur_path)
        list_temp.remove("VERSION")
        dcm_list = [os.path.join(cur_path,f) for f in list_temp]
        # Load sample
        first = pydicom.dcmread(dcm_list[0]).pixel_array
        sahpe = first.shape
        org_dbt = np.zeros([sahpe[0], sahpe[1], len(dcm_list)])
        for l in range(len(dcm_list)):
            org_dbt[:,:,l] = pydicom.dcmread(dcm_list[l]).pixel_array
        # preprocessing
        preprocess_dbt = preprocess_3D(org_dbt, self.dim[0], self.dim[1], self.dim[2])
        X[i,:, :, :] = preprocess_dbt.preprocessing()
        # Load labels       
        y[i,1] = list_labels_temp[i]
        y[i,0] = 1-list_labels_temp[i]
        
    return X, y



