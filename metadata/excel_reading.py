# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:39:10 2022

@author: Idan
"""
            
import pandas as pd
import os
import pydicom
import numpy as np


def create_tomo_excel(new_excel_name, data_dir_name,data_path):
    if os.path.isfile(os.path.join(data_path, new_excel_name)):
            print('File already exists!')
    else:
        df = pd.read_excel(os.path.join(data_path,'SOROKA_DBT_US_Metadata.xlsx'))
        dicom_excel_df = pd.DataFrame(columns = ['Name', 'Age', 'Bi-Rads', 'Laterality', 'View', 'Mass', 'Modality', 'Images Path'])
        
        # get patients names list
        #names = df['Name'].dropna().unique()
        names = os.listdir(os.path.join(data_path,data_dir_name))
        # patient loop
        for patient_name in names:
            print(patient_name)
            if patient_name=='AH8069' or patient_name=='VM0156' or patient_name=='TE0788_1' or patient_name=='GE6237':
                continue
            patient_path = os.path.join(data_path,data_dir_name,patient_name,'TOMO')
            subfolder_path = [os.path.join(patient_path, o) for o in os.listdir(patient_path) 
                              if os.path.isdir(os.path.join(patient_path,o))]    
            for sub_path in subfolder_path:
                all_images_path = [os.path.join(sub_path, o) for o in os.listdir(sub_path) 
                            if os.path.isdir(os.path.join(sub_path,o))]  
                for path in all_images_path:
                    if len(os.listdir(path))>10:
                        # get data from dicom file                    
                        ds = pydicom.dcmread(os.path.join(path,os.listdir(path)[0]))
                        age = ds['PatientAge'].value[:-1]
                        laterality = ds['5200', '9229'][0]['0020','9071'][0]['0020','9072'].value
                        view_full = ds['0054', '0220'][0]['0008','0104'].value
                        if view_full == 'cranio-caudal':
                            view = 'CC'
                        elif view_full == 'medio-lateral oblique':
                            view = 'MLO'
                        # Labels extraction
                        patient_df = df.loc[df['Name'] == patient_name]
                        relevant_row = patient_df.loc[np.logical_and(patient_df['Laterality']==laterality, patient_df['Modality']=='Tomo')]
                        mass = relevant_row['Detected Mass'].iloc[0]
                        birad = relevant_row['Bi-Rads'].iloc[0]
                        # create data frame
                        dicom_excel_df = dicom_excel_df.append({'Name': patient_name, 'Age': age, 'Bi-Rads':birad, 'Laterality': laterality, 
                                'View': view, 'Mass': mass, 'Modality': 'DBT', 'Images Path':path}, ignore_index=True)
                print(patient_name, '  -  Done!')
        # save to excel
        dicom_excel_df.to_excel(new_excel_name)
                    
                    
#%% main - DBT
#data_path = '/mnt/md0/databases/DBT_US_Soroka'
data_path = '/workspace/DBT_US_Soroka'
data_dir_name = 'Anonymouse Data'
new_excel_name = 'Soroka_DBT_Metadata_Dicom.xlsx'          
create_tomo_excel(new_excel_name,data_dir_name,data_path)





