import os
import pandas as pd
import pydicom
import re
import numpy as np

data_annotated_list = os.listdir('/workspace/DBT_US_Soroka/Manually_Selected_Frames/Positive')
data_check_list = os.listdir('/workspace/DBT_US_Soroka/Manually_Selected_Frames/Check/P')

# get scans names - need to check
uniqe_filename_check = []
for name in data_check_list:
    if name[:12] not in uniqe_filename_check:
        uniqe_filename_check.append(name[:12])

scans_names = []
scans_view = []
scans_laterality = []
all_frames = []

for name in uniqe_filename_check:
    scans_names.append(name[:6])
    scans_laterality.append(name[7])
    view = name[9:12]
    if view[-1]=='_':
        view = view[:-1]
    scans_view.append(view)
    all_frames.append([])

# get study date
dir = '/workspace/DBT_US_Soroka/Anonymouse_Data'
study_date =[]
series_date = []
for n in scans_names:
    cur_path = os.path.join(dir,n,'TOMO')
    sub_dir = os.listdir(cur_path)
    sub_dir.remove('DICOMDIR')
    sub_dir.remove('LOCKFILE')
    sub_dir.remove('VERSION')
    sub_dir = sub_dir[0]
    ss_dir = os.listdir(os.path.join(cur_path,sub_dir))
    ss_dir.remove('VERSION')
    ss_dir=ss_dir[0]
    file_name = os.listdir(os.path.join(cur_path,sub_dir, ss_dir))
    file_name.remove('VERSION')
    final_path = os.path.join(cur_path,sub_dir, ss_dir, file_name[0])
    dcm = pydicom.dcmread(final_path)
    study_date.append(dcm[0x0008,0x0020].value)
    series_date.append(dcm[0x0008,0x0021].value)


# create excel
df_list = pd.DataFrame({"study_ID": scans_names,
                        "View": scans_view,
                        "Laterality": scans_laterality,
                        "Study_Date": study_date,
                        "Series_Date": series_date})
df_list.to_csv("scans_to_check_tomo.csv", index=False)

# ALL SCANS IN STUDY
uniqe_filename_annotated = []
for name in data_annotated_list:
    if name[:12] not in uniqe_filename_annotated:
        uniqe_filename_annotated.append(name[:12])


s = pd.Series(data_annotated_list)
for name in uniqe_filename_annotated:
    list_frames = []
    res = [i for i, val in enumerate(s.str.contains(name)) if val]
    sub_list = [data_annotated_list[i] for i in np.array(res)]
    for sub_name in sub_list:
        frame = int(re.findall(r'\d+', sub_name)[-1])
        list_frames.append(frame)
    all_frames.append(list_frames)


for name in uniqe_filename_annotated:
    scans_names.append(name[:6])
    scans_laterality.append(name[7])
    view = name[9:12]
    if view[-1]=='_':
        view = view[:-1]
    scans_view.append(view)

print(len(scans_names))
print(all_frames)
df_list_all = pd.DataFrame({"study_ID": scans_names,
                        "View": scans_view,
                        "Laterality": scans_laterality,
                        "Selected_Frames": all_frames})
df_list_all.to_csv("annotations_tomo.csv", index=False)

