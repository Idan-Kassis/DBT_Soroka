
import pandas as pd

# Read the Excel file into a DataFrame
df = pd.read_excel('/workspace/DBT_US_Soroka/Codes/DBT/2D/tomo_subjects_sets_all_2804.xlsx')

# Combine all columns into a single list
all_data_list = df.values.flatten().tolist()
filtered_data_list = [x for x in all_data_list if not pd.isna(x)]

# read birads information
df = pd.read_excel('/workspace/DBT_US_Soroka/Codes/DBT/Soroka_DBT_Metadata_Dicom.xlsx')

condition = df['Name'].isin(filtered_data_list)
filtered_df = df[condition]['Bi-Rads'].tolist()
modified_list = ["4" if x in ["4A", "4B", "4C"] else x for x in filtered_df]

int_list = [int(x) for x in modified_list]
count_dict = {}

# Count the occurrences of each number
for num in int_list:
    if num in count_dict:
        count_dict[num] += 1
    else:
        count_dict[num] = 1

# Print the count of each number
numbers = []
counting = []

for num, count in count_dict.items():
    numbers.append(num)
    counting.append(count)

pairs = list(zip(numbers, counting))
sorted_pairs = sorted(pairs, key=lambda x: x[0])
sorted_list1, sorted_list2 = zip(*sorted_pairs)
print("Sorted list1:", list(sorted_list1))
print("Sorted list2:", list(sorted_list2))


import matplotlib.pyplot as plt
import numpy as np

y = np.array(sorted_list2)
mylabels = ["0", "1", "2", "3", "4", "5" , "6"]

plt.pie(y, labels = mylabels, autopct='%1.1f%%',pctdistance=1.25, labeldistance=.6)
plt.savefig('birads_pie_chart.jpg')
plt.close()
plt.clf()
