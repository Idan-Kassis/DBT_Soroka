# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:49:28 2022

@author: Idan
"""
import os
import pydicom
import pymedphys
import numpy as np 


class anonimization():
    def __init__(self, path_data = 'Data'):
        self.path_data = path_data
        
        
    
    def save_anonymouse_dicom(self, dcm_file_anonymouse, name):
        pydicom.filewriter.dcmwrite(name, dcm_file_anonymouse)
    
    def read_dicom(self, name):
        # read files
        ds = pydicom.dcmread(name)
        return ds
        
    def get_dicom_names(self):
        files = []
        dir_path = os.path.join(os.getcwd(), self.path_data)
        for (dir_path, dir_names, file_names) in os.walk(dir_path): 
            if file_names == []:
                continue
            if dir_names == []:
                for name in file_names:
                    files.append(os.path.join(dir_path,name))
                
        # remove the path
        only_names = []
        for f in files:
            only_names.append(os.path.basename(f))
        # Remove irrelevant files
        to_remove = ['DICOMDIR', 'LOCKFILE', 'VERSION']
        for remove_word in to_remove:
            indices = [index for index, element in enumerate(only_names) if element == remove_word]
            indices = np.array(indices, int)
            for del_idx in indices:
                del files[del_idx]
                del only_names[del_idx]
                indices -=1
        self.files = files
        return files
        
    def make_anonymouse(self):
        for name in self.files:
            dcm_file = self.read_dicom(name)
            
            # Anonymise
            dcm_file_anonymouse = pymedphys.dicom.anonymise(dcm_file, replace_values=True, 
                                      keywords_to_leave_unchanged=(["PatientAge", "PatientSex", "StudyDate", "SeriesDescription"]), 
                                      delete_private_tags=True, 
                                      delete_unknown_tags=None, 
                                      copy_dataset=True, 
                                      replacement_strategy=None, 
                                      identifying_keywords=None)
            print(name, '  -  Done!')
            self.save_anonymouse_dicom(dcm_file_anonymouse, name)
        
        
# ---------------------------------- Main -------------------------------------
# The code need to run from 'D:\SOROKA_DBT_US_Database'
dicom_anonimization = anonimization()
dicom_file_names = dicom_anonimization.get_dicom_names()
dicom_anonimization.make_anonymouse()
        
        
