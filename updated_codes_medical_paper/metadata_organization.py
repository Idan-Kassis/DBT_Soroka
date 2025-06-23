import pandas as pd
import os
import pydicom
import numpy as np


def create_tomo_excel(new_excel_name, data_path):
    if os.path.isfile(os.path.join(data_path, new_excel_name)):
        print('File already exists!')
    else:
        dicom_excel_df = pd.DataFrame(
            columns=['Name', 'Age', 'Laterality', 'View', 'Modality', 'Images Path'])

        # get patients names list
        names = os.listdir(data_path)
        # patient loop
        for patient_name in names:
            print(patient_name)
            try:
                patient_path = os.path.join(data_path, patient_name)
                subfolder_path = [os.path.join(patient_path, o) for o in os.listdir(patient_path)
                                  if os.path.isdir(os.path.join(patient_path, o))]
                for sub_path in subfolder_path:
                    all_images_path = [os.path.join(sub_path, o) for o in os.listdir(sub_path)
                                       if os.path.isdir(os.path.join(sub_path, o))]
                    for path in all_images_path:
                        if len(os.listdir(path)) > 10:
                            # get data from dicom file
                            ds = pydicom.dcmread(os.path.join(path, os.listdir(path)[0]))
                            age = ds['PatientAge'].value[:-1]
                            laterality = ds['5200', '9229'][0]['0020', '9071'][0]['0020', '9072'].value
                            view_full = ds['0054', '0220'][0]['0008', '0104'].value
                            if view_full == 'cranio-caudal':
                                view = 'CC'
                            elif view_full == 'medio-lateral oblique':
                                view = 'MLO'
                            # create data frame
                            dicom_excel_df.loc[len(dicom_excel_df)] = {
                                'Name': patient_name,
                                'Age': age,
                                'Laterality': laterality,
                                'View': view,
                                'Modality': 'DBT',
                                'Images Path': path
                            }
                    print(patient_name, '  -  Done!')
            except:
                print('error')
                continue
        # save to excel
        dicom_excel_df.to_excel(new_excel_name)


# %% main - DBT
# data_path = '/mnt/md0/databases/DBT_US_Soroka'
data_path = '/Volumes/Elements/negative'
new_excel_name = 'Soroka_DBT_Metadata_Dicom.xlsx'
create_tomo_excel(new_excel_name, data_path)