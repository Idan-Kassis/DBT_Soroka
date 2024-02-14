import numpy as np
import os
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image

data_path = "/workspace/DBT_US_Soroka/Preprocessed_2D"
data_save_path = "/workspace/DBT_US_Soroka/Preprocessed_2D/Segmented"

sets = ['Train', 'Val', 'Test']
classes = ['Negative', 'Positive']

for set in sets:
    for label in classes:
        list_img = os.listdir(os.path.join(data_path, set, label))
        path_list = [os.path.join(os.path.join(data_path, set, label), o) for o in list_img]
        for file in path_list:
            image = cv2.imread(file)
            im = image/1023.
            rounded = np.zeros(im.shape)
            rounded[im>=0.1]=1
            result = np.where(rounded == 1.)
            segmented = image[np.min(result[0]):np.max(result[0]), np.min(result[1]):np.max(result[1])]
            filename = os.path.join(data_save_path, set, label, os.path.basename(file))
            cv2.imwrite(filename, segmented)
            print(os.path.basename(file), '  -  Done!')