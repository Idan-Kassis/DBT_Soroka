# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:50:20 2022

@author: Dror
"""
import pandas as pd
import pydicom
import os
import numpy as np
from clearml import Task

Task.init(project_name='DBT-SOROKA', task_name='Validiy_Check')


df = pd.read_excel('Soroka_DBT_Metadata_Dicom.xlsx')
path_list = df['Images Path']

for cur_path in path_list:
    dcm_list = os.listdir(cur_path)
    dcm_list.remove("VERSION")
    dcm_path_list = [os.path.join(cur_path,f) for f in dcm_list]
    try:
        first = pydicom.dcmread(dcm_path_list[0]).pixel_array
        sahpe = first.shape
        dbt = np.zeros([sahpe[0], sahpe[1], len(dcm_path_list)])
        for l in range(len(dcm_path_list)):
            dbt[:,:,l] = pydicom.dcmread(dcm_path_list[l]).pixel_array
    except: print('Reading Error!   -----   ', dcm_path_list[l])
    
print('Done !')
        
        