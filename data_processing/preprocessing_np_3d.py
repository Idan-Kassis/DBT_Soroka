# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:58:05 2022

@author: Dror
"""

from scipy import ndimage
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from clearml import Task

Task.init(project_name='Preprocessing-DBT-SOROKA', task_name='Preprocessing')

# Preprocessing for 3D
class preprocess_3D():
    
    def __init__(self, image, desired_height, desired_width, desired_depth):
        self.image = image
        self.min = 0
        self.max = 65535
        self.desired_depth = desired_depth
        self.desired_width = desired_width
        self.desired_height = desired_height
        
    
    def get_crop_idx(self, img, min_size = 1000):
    # =============================================================================
    #         crop black areas and leave only breast parts
    # =============================================================================
        # convert img into uint8 and make it binary for threshold value of 1.    
        _,thresh = cv2.threshold(img.astype('uint8'), 0, 255, cv2.THRESH_BINARY)
        #find all the connected components (white blobs in the image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        # take out the background (found also as a blob)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
    
        # get rid of small blobs
        img2 = np.zeros((output.shape))
        #for every component in the image, we keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
                
        # find contours and bounding rectangle
        contours, _ = cv2.findContours(img2.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        
        return [y, x, y+h, x+w]
    
    def crop_3D_img(self, img):
        # initialization
        y = img.shape[0]
        x = img.shape[1]
        y_h = 0
        x_w = 0
        
        for i in range(img.shape[-1]):
            im = cv2.convertScaleAbs(img[:,:,i], alpha=(255.0/65535.0))
            plt.imshow(im)
            [y1, x1, y_h1, x_w1] = self.get_crop_idx(im)
            if x1 < x : x = x1
            if y1 < y : y = y1
            if y_h1 > y_h : y_h = y_h1
            if x_w1 > x_w : x_w = x_w1
        
        cropped = self.image[y:y_h, x:x_w, :]
        return cropped


    def normalize(self, img):
        # Normalize the volume
        img[img < self.min] = self.min
        img[img > self.max] = self.max
        img = (img - self.min) / (self.max - self.min)
        img = img.astype("float32")
        return img
        
    def resize(self, img):
        # Set the desired depth
        # Get current depth
        current_depth = img.shape[2]
        current_width = img.shape[1]
        current_height = img.shape[0]
        # Compute depth factor
        depth = current_depth / self.desired_depth
        width = current_width / self.desired_width
        height = current_height / self.desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        # Rotate
        #img = ndimage.rotate(img, 90, reshape=False)
        # Resize across z-axis
        img = ndimage.zoom(img, (height_factor, width_factor, depth_factor), order=1)
        return img
        
    def preprocessing(self):
        
        img = self.image
        # cropping
        image = self.crop_3D_img(img)
        # Normalize
        image = self.normalize(image)
        # Resize width, height and depth
        image = self.resize(image)
        return image

#%% 
data_path = '/workspace/DBT_US_Soroka/Numpy Data'
save_path = '/workspace/DBT_US_Soroka/Preprocessed Data'
classes_names = ['Negative', 'Positive']
input_size = [200,200,30]

for name in classes_names:
    # Get numpy foiles paths
    class_path = os.path.join(data_path, name)
    list_names = os.listdir(class_path)
    all_paths = [os.path.join(class_path,f) for f in list_names]
    if name == 'Negative': continue
    if name == 'Positive': all_paths = all_paths[589:]
    for file in all_paths:
        # Read file
        img = np.load(file)
        # Preprocessing
        pp = preprocess_3D(img, input_size[0], input_size[1], input_size[2])
        try:
            pp_img = pp.preprocessing()
            np.save(os.path.join(save_path, name, os.path.basename(file)), pp_img)
            del img, pp_img
        except:
            print('Preprocessing Failed   -   ', file)
    
    
    


