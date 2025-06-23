import pandas as pd
import re
import os
import pydicom

def get_age(predictions_df, age_df, data_path="/Volumes/Elements/negative") -> pd.DataFrame:
    age_df["Filename"] = age_df["Name"].astype(str) + "_" + age_df["Laterality"].astype(str)
    age_dict = age_df.set_index("Filename")["Age"].to_dict()
    predictions_df["Age"] = predictions_df["Filename"].map(age_dict)

    missing_ages = predictions_df[predictions_df["Age"].isna()]["Filename"].str.split("_").str[0].unique()
    for patient_name in missing_ages:
        patient_path = os.path.join(data_path, patient_name)
        if not os.path.exists(patient_path):
            continue
        subfolder_paths = [os.path.join(patient_path, o) for o in os.listdir(patient_path)
                           if os.path.isdir(os.path.join(patient_path, o))]
        for sub_path in subfolder_paths:
            all_images_paths = [os.path.join(sub_path, o) for o in os.listdir(sub_path)
                                if os.path.isdir(os.path.join(sub_path, o))]
            for path in all_images_paths:
                if len(os.listdir(path)) > 10:
                    try:
                        ds = pydicom.dcmread(os.path.join(path, os.listdir(path)[0]))
                        age = ds['PatientAge'].value[:-1]
                        predictions_df.loc[predictions_df["Filename"].str.startswith(patient_name), "Age"] = age
                        break
                    except Exception as e:
                        print(f"Error reading DICOM for {patient_name}: {e}")
    return predictions_df


def get_biradscore_and_size(predictions_df, metadata_df) -> pd.DataFrame:
    metadata_df["Filename"] = metadata_df["Name"] + "_" + metadata_df["Laterality"]
    birad_dict = metadata_df.set_index("Filename")["Bi-Rads"].to_dict()
    size_dict = metadata_df.set_index("Filename")["Tumor Size"].to_dict()
    predictions_df["Bi-Rads"] = predictions_df["Filename"].map(birad_dict)
    predictions_df["Tumor Size"] = predictions_df["Filename"].map(size_dict)
    return predictions_df


def get_birads_density(predictions_df) -> pd.DataFrame:
    density_df_new = pd.read_excel("/Users/idankassis/Desktop/Thesis/medical_paper/sampled_events_negative.xls")
    density_df_old = pd.read_csv("/Users/idankassis/Desktop/Thesis/Paper/revision/results_test_case-based_with_density.csv")

    # Extract tissue type after "אפיון כללי של רקמת השד:"
    density_dict = {}
    for _, row in density_df_new.iterrows():
        match_tissue = re.search(r"איפיון כללי של רקמת השד:\s*(.*)", str(row["Result"]))
        if match_tissue:
            tissue_type = match_tissue.group(1).strip()
            filename_base = str(row["coding"])
            density_dict[filename_base + "_L"] = tissue_type
            density_dict[filename_base + "_R"] = tissue_type
    birads_summary_dict = {}
    for _, row in density_df_new.iterrows():
        match_left = re.search(r"שד שמאל:\s*BIRADS (\d+)", str(row["Result"]))
        match_right = re.search(r"שד ימין:\s*BIRADS (\d+)", str(row["Result"]))
        filename_base = str(row["coding"])
        if match_left:
            birads_summary_dict[filename_base + "_L"] = match_left.group(1)
        if match_right:
            birads_summary_dict[filename_base + "_R"] = match_right.group(1)


    density_dict.update(density_df_old.set_index("Filename")["Density"].to_dict())
    predictions_df["BIRADS_Density"] = predictions_df["Filename"].map(density_dict)
    predictions_df["Bi-Rads"].update(predictions_df["Filename"].map(birads_summary_dict))
    return predictions_df


def get_biopsy(predictions_df, biopsy_df) -> pd.DataFrame:
    biopsy_df["Filename"] = biopsy_df["Name"] + "_" + biopsy_df["Laterality"]
    biopsy_dict = biopsy_df.set_index("Filename")["Biopsy_Result"].to_dict()
    predictions_df["Biopsy_Result"] = predictions_df["Filename"].map(biopsy_dict)
    predictions_df["true_label_biopsy"] = predictions_df["true_labels"]
    predictions_df.loc[predictions_df["Biopsy_Result"].notna(), "true_label_biopsy"] = 1
    return predictions_df



if __name__ == '__main__':
    df=pd.read_csv('/Users/idankassis/Desktop/Thesis/codes/pythonProject1/predictions/1024-res/results_test_case-based_1024.csv')
    subgroups_df = df.copy()

    biopsy_df = pd.read_csv('/Users/idankassis/Desktop/Thesis/metadata/biopsy_results_Final.csv')
    subgroups_df = get_biopsy(subgroups_df, biopsy_df)
    metadata = pd.read_excel('/Users/idankassis/Desktop/Thesis/metadata/SOROKA_DBT_US_Metadata.xlsx')
    subgroups_df = get_biradscore_and_size(subgroups_df, metadata)
    age_df = pd.read_excel('/Users/idankassis/Desktop/Thesis/metadata/Soroka_DBT_Metadata_Dicom.xlsx')
    subgroups_df = get_age(subgroups_df, age_df)
    subgroups_df = get_birads_density(subgroups_df)
    subgroups_df.to_csv('/Users/idankassis/Desktop/Thesis/medical paper - 1024/subgroups.csv')
