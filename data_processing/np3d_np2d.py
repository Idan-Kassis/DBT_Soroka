import numpy as np
import os
from clearml import Task
Task.init(project_name='DBT-SOROKA-numpy2d', task_name='3d->2d')

classes = ['Negative', 'Positive']

for class_name in classes:
    if class_name=='Negative': continue
    path = os.path.join('/workspace/DBT_US_Soroka/Numpy_Data', class_name)
    list_np = [os.path.join(path, o) for o in os.listdir(path)]
    list_np = list_np[800:]
# saving loop loop
    for sample in list_np:
        img = np.load(sample)
        for slice_idx in range(img.shape[-1]):
            slice = img[:,:,slice_idx]
            new_file_name = os.path.join('/workspace/DBT_US_Soroka/Numpy_Data_2D', class_name, os.path.basename(sample)[0:-4] +'_'+ str(slice_idx))
            np.save(new_file_name, slice)
        print(os.path.basename(sample)[0:-4],'  -  Done!')