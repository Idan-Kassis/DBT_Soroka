
import os
import pandas as pd
import numpy as np

data_path = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384"
sets = ["Train", "Val", "Test"]
labels = ["Positive", "Negative"]

df = pd.read_excel("/workspace/DBT_US_Soroka/Codes/DBT/Soroka_DBT_Metadata_Dicom.xlsx")


for l in labels:
    uniqe_scans = []
    uniqe_cases = []
    for s in sets:
        path = os.path.join(data_path, s, l)
        files = os.listdir(path)
        for name in files:
            if name[:8] not in uniqe_cases:
                uniqe_cases.append(name[:8])
            if name[:12] not in uniqe_scans:
                uniqe_scans.append(name[:12])
    print(l, '  -  ', len(uniqe_scans), '  scans')
    print(l, '  -  ', len(uniqe_cases), '  cases')
    
    ages = []
    for case in uniqe_cases:
        age = df[df["Name"]==case[:-2]]["Age"].values[0]
        ages.append(age)
        
    mean = np.mean(np.asarray(ages))
    std = np.std(np.asarray(ages))
    print(l,'  age - ',mean,' +- ', std)
    print(l,' - ', np.min(np.asarray(ages)))
name_list = []
for case in uniqe_cases:
    name_list.append(case[:-2])


# slice numbers
csv_file = pd.read_excel("/workspace/DBT_US_Soroka/Codes/DBT/Soroka_DBT_Metadata_Dicom.xlsx")
df = csv_file[csv_file['Name'].isin(name_list)]
paths = df["Images Path"]

scan_len = []
for path in paths:
    path = path[:35] + '_' + path[36:]
    scan_len.append(len(os.listdir(path))-1)
arr = np.array(scan_len)
print('Minimum slices - ', np.partition(arr, 2)[2])
print('Maximum slices - ', np.partition(arr, -3)[-3])
print('Mean slices - ', np.mean(arr))
print('STD slices - ', np.std(arr))

# positive scans - train
for l in labels:
    uniqe_scans = []
    path = os.path.join(data_path, "Train", l)
    files = os.listdir(path)
    for name in files:
        if name[:12] not in uniqe_scans:
            uniqe_scans.append(name[:12])
    print(l, '  -  ', len(uniqe_scans), '  scans')



