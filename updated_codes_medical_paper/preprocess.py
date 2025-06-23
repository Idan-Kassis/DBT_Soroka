import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydicom
from scipy import ndimage
import cv2
import os
from PIL import Image

class preprocess_2D():

    def __init__(self, image, desired_height, desired_width):
        self.image = image
        self.min = 0
        self.max = 65535
        self.desired_width = desired_width
        self.desired_height = desired_height

    def resize(self, img):
        dim = (self.desired_width, self.desired_height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    def preprocessing(self):
        img = self.image
        image = self.resize(img)
        return image


def read_3d_dicom(path):
    list_temp = os.listdir(path)
    list_temp.remove("VERSION")
    dcm_list = [os.path.join(path, f) for f in list_temp]
    first = pydicom.dcmread(dcm_list[0]).pixel_array
    sahpe = first.shape
    del first
    DBT = np.zeros([sahpe[0], sahpe[1], len(dcm_list)])
    for l in range(len(dcm_list)):
        DBT[:, :, l] = pydicom.dcmread(dcm_list[l]).pixel_array
    return DBT


# %% --------------------------- Data Reading ---------------------------------

# Excel Data Path
df = pd.read_excel("Soroka_DBT_Metadata_Dicom.xlsx")
save_path = '/Users/idankassis/Desktop/Thesis/preprocessed_negative_1016-end'
os.makedirs(save_path, exist_ok=True)
# Loop - Read the data - till len(df)
for i in range(1016, len(df)):
    print(i)
    subject = df.iloc[i]
    patient, view, laterality = subject["Name"], subject["View"], subject["Laterality"]
    image_path = subject["Images Path"].replace("Elements", "Elements 1")

    list_temp = os.listdir(image_path)
    list_temp.remove("VERSION")
    dcm_list = [os.path.join(image_path, f) for f in list_temp]

    for slice_idx in range(len(dcm_list)):
        file_name = f'{patient}_{laterality}_{view}_{slice_idx}.png'
        if os.path.exists(os.path.join(save_path, file_name)):
            continue
        try:
            slice = pydicom.dcmread(dcm_list[slice_idx]).pixel_array

            im = slice / 1023.
            rounded = np.zeros(im.shape)
            rounded[im >= 0.1] = 1
            result = np.where(rounded == 1.)
            segmented = slice[np.min(result[0]):np.max(result[0]), np.min(result[1]):np.max(result[1])]
            input_size = segmented.shape
            pp = preprocess_2D(segmented, input_size[0], input_size[1])
            pp_img = pp.preprocessing()
            rescaled_image = (np.maximum(pp_img, 0) / pp_img.max()) * 255
            final_image = Image.fromarray(np.uint8(rescaled_image))
            final_image.save(os.path.join(save_path, file_name))

            print(file_name, '  -   Done!')
        except:
            print(f'error reading slice - {file_name}')
            continue
