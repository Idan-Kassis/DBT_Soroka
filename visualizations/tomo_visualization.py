import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2

def create_video(file_name, save_path):
    array_3d = np.load(file_name)
    array_3d = array_3d.astype(np.float32)

    # get original array shape and calculate new shape
    original_shape = array_3d.shape
    print(original_shape)

    # define video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 15, (original_shape[1], original_shape[0]))

    # loop through frames
    for i in range(original_shape[2]):
        # extract a single frame from 3D array
        frame = array_3d[:, :, i] / 1023.

        # convert grayscale to color
        # resized_frame = cv2.convertScaleAbs(resized_frame)
        resized_frame = (frame * 255).astype(np.uint8)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
        # segmentation

        # write the resized frame to output video
        out.write(resized_frame)

    # release video object
    out.release()


path = "/workspace/DBT_US_Soroka/Codes/DBT/visualization"
files = ["AF6848_R_MLO.npy"]

flag = True
if flag:
    for file in files:
        save_path = os.path.join(path, os.path.splitext(file)[0] + ".avi")
        create_video(os.path.join(path, file), save_path)

import pydicom
from PIL import Image


def convert_dcm_to_png(dcm_file_path, output_png_path):
    # Read the DICOM file
    dcm_data = pydicom.dcmread(dcm_file_path)

    # Extract pixel data
    pixel_array = dcm_data.pixel_array

    # Normalize pixel values to the range [0, 255]
    normalized_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255

    # Convert to 8-bit unsigned integer
    image_array = normalized_array.astype('uint8')

    # Create PIL image
    image = Image.fromarray(image_array)

    # Save as PNG
    image.save(output_png_path)


# Usage
dcm_file_path = os.path.join(path, "I2600000")
output_png_path = os.path.join(path,"I2600000.png")

convert_dcm_to_png(dcm_file_path, output_png_path)
