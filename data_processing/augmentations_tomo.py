
from volumentations import Compose, Rotate, Flip, RandomRotate90, GaussianNoise, RandomGamma
import numpy as np
import os
from clearml import Task
Task.init(project_name='DBT-SOROKA-Augmentations', task_name='Augmentations')

# Define parameters
num_augmentations = 50
image_shape = [200,200,30]
classes = ['Negative', 'Positive']

# Augmentation function
def get_augmentation(patch_size):
    return Compose([
        Rotate((-10, 10), (0, 0), (0, 0), p=0.5),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        RandomRotate90((0, 1), p=0.5),
    ], p=1.0)

# Train cancer IDs
for class_name in classes:
    path = os.path.join('/workspace/DBT_US_Soroka/Preprocessed Data', class_name)
    list_train = [os.path.join(path, o) for o in os.listdir(path)]  
    list_train = list_train[:round(0.6*len(list_train))]

# Augmentation loop
    for sample in list_train:
        img = np.load(sample)
        data = {'image': img}
        for aug_idx in range(num_augmentations):
            aug = get_augmentation((image_shape[0], image_shape[1], image_shape[2]))
            aug_data = aug(**data)
            img_augmented = aug_data['image']
            new_file_name = os.path.join('/workspace/DBT_US_Soroka/Augmented Data', class_name, os.path.basename(sample)[0:-4] +'_'+ str(aug_idx))
            np.save(new_file_name, img_augmented)