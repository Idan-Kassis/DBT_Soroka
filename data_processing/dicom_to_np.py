
# %% -------------------------- Initialization --------------------------------
# Libraries

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydicom

def read_3d_dicom(path):
    list_temp = os.listdir(path)
    list_temp.remove("VERSION")
    dcm_list = [os.path.join(path,f) for f in list_temp]
    # Load sample
    first = pydicom.dcmread(dcm_list[0]).pixel_array
    sahpe = first.shape
    del first
    DBT = np.zeros([sahpe[0], sahpe[1], len(dcm_list)])
    for l in range(len(dcm_list)):
        DBT[:,:,l] = pydicom.dcmread(dcm_list[l]).pixel_array
    return DBT

# %% --------------------------- Data Reading ---------------------------------

# Excel Data Path
df = pd.read_excel("Soroka_DBT_Metadata_Dicom.xlsx")
# Remove invalid data
invalid_list = ["/workspace/DBT_US_Soroka/Anonymouse Data/KY2807/TOMO/WSCSFUVQ/52YQVNYJ",
                "/workspace/DBT_US_Soroka/Anonymouse Data/AA5258/TOMO/AHFQTQZB/0NVO053E",
                "/workspace/DBT_US_Soroka/Anonymouse Data/RI0924/TOMO/V3MCX5J4/15THUGIA",
                "/workspace/DBT_US_Soroka/Anonymouse Data/RY1163/TOMO/EVTUTFMB/LM3UH54S"]
for invalid in invalid_list:
    invalid_loc = df.loc[df['Images Path']==invalid].index
    df = df.drop(invalid_loc[0],axis=0)
    
# Loop - Read the data
for i in range(len(df)):
    subject = df.iloc[i]
    patient , view, laterality = subject["Name"], subject["View"], subject["Laterality"]
    image_path = subject["Images Path"]
    image_3d = read_3d_dicom(image_path)
    
    if subject["Mass"] == 'P':
        label = 'Positive'
    else: label = 'Negative'
        
    # Save to numpy array
    np.save(os.path.join('/workspace/DBT_US_Soroka/Numpy Data', label ,str(patient+'_'+laterality+'_'+view)), image_3d)
        
    print(patient)
