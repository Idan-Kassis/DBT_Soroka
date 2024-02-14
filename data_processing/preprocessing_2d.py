from scipy import ndimage
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from clearml import Task
from PIL import Image

Task.init(project_name='Preprocessing-DBT-2D-SOROKA', task_name='Preprocessing')


# Preprocessing for 3D
class preprocess_2D():

    def __init__(self, image, desired_height, desired_width):
        self.image = image
        self.min = 0
        self.max = 65535
        self.desired_width = desired_width
        self.desired_height = desired_height

    def get_crop_idx(self, img, min_size=1000):
        # =============================================================================
        #         crop black areas and leave only breast parts
        # =============================================================================
        # convert img into uint8 and make it binary for threshold value of 1.
        _, thresh = cv2.threshold(img.astype('uint8'), 0, 255, cv2.THRESH_BINARY)
        # find all the connected components (white blobs in the image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        # take out the background (found also as a blob)
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        # get rid of small blobs
        img2 = np.zeros((output.shape))
        # for every component in the image, we keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255

        # find contours and bounding rectangle
        contours, _ = cv2.findContours(img2.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        return [y, x, y + h, x + w]

    def crop_img(self, img):
        # initialization
        im = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
        [y, x, y_h, x_w] = self.get_crop_idx(im)
        cropped = img[y:y_h, x:x_w]
        return cropped

    def normalize(self, img):
        # Normalize the volume
        norm_img = cv2.normalize(img, None, 0., 1., cv2.NORM_MINMAX)
        # img[img < self.min] = self.min
        # img[img > self.max] = self.max
        # img = (img - self.min) / (self.max - self.min)
        # img = img.astype("float32")
        return norm_img

    def resize(self, img):
        dim=(self.desired_width, self.desired_height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    def preprocessing(self):

        img = self.image
        # cropping
        #image = self.crop_img(img)
        # Normalize
        #image = self.normalize(image)
        # Resize width, height and depth
        image = self.resize(img)
        return image


# %%
data_path = '/workspace/DBT_US_Soroka/Numpy_Data_2D'
save_path = '/workspace/DBT_US_Soroka/Preprocessed_2D'
classes_names = ['Negative', 'Positive']
input_size = [800,800]

for name in classes_names:
    if name=='Negative': continue
    # Get numpy foiles paths
    class_path = os.path.join(data_path, name)
    list_names = os.listdir(class_path)
    all_paths = [os.path.join(class_path, f) for f in list_names]
    for file in all_paths[20890:]:
        # Read file
        img = np.load(file)
        input_size = img.shape
        # Preprocessing
        pp = preprocess_2D(img, input_size[0], input_size[1])
        #try:
        pp_img = pp.preprocessing()
        rescaled_image = (np.maximum(pp_img, 0) / pp_img.max()) * 255
        final_image = Image.fromarray(np.uint8(rescaled_image))
        final_image.save(os.path.join(save_path, name, os.path.basename(file)).replace('npy','png'))

        #np.save(os.path.join(save_path, name, os.path.basename(file)), np.uint8(pp_img))
        del img, pp_img
        print(os.path.basename(file), '  -   Done!')
        #except:
        #    print('Preprocessing Failed   -   ', file)
