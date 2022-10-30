# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 09:27:33 2022

@author: Idan
"""
import cv2 
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt


def crop_image(img, min_size = 1000):
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
    
    
    ## crop the original image
    #crop = img[y:y+h,x:x+w] 

    #return crop
    return [y, x, y+h, x+w]

#%%


dcm_list = os.listdir(os.getcwd())[:-1]
[rows , cols] = pydicom.dcmread(dcm_list[0]).pixel_array.shape
dbt_3d = np.zeros([rows, cols, len(dcm_list)])
# initialization
y = rows
x = cols
y_h = 0
x_w = 0

for i, name in enumerate(dcm_list):
    ds = pydicom.dcmread(name)
    img = cv2.convertScaleAbs(ds.pixel_array, alpha=(255.0/65535.0))
    [y1, x1, y_h1, x_w1] = crop_image(img)
    if x1 < x : x = x1
    if y1 < y : y = y1
    if y_h1 > y_h : y_h = y_h1
    if x_w1 > x_w : x_w = x_w1
    dbt_3d[:,:,i] = ds.pixel_array

dbt_3d = dbt_3d[y:y_h, x:x_w, :]







