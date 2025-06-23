from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np

df = pd.read_csv(str("negative_test_prediction-1016-end.csv"))
file_names = df["Filename"]

# get the subjects and view list
uniqe_filename = []
for name in file_names:
    if name[:12] not in uniqe_filename:
        uniqe_filename.append(name[:12])

probabilities = []
true_labels = []
ten_slice_prob = []
file_names = df["Filename"]
for name in uniqe_filename:
    res = file_names.str.contains(pat=name)
    sorted_list = sorted(zip(list(df["Filename"][res.values]), list(df["Probabilities"][res.values])), key=lambda x: int(x[0].split("_")[-1].split(".")[0]))

    if len(res) < 20:
        print('too short!')
        continue
    sorted_list_probs = [item[1] for item in sorted_list]
    relevant_probs = np.array(sorted_list_probs)
    y_true = df["true_labels"][res.values]
    probabilities.append(np.median(relevant_probs))
    true_labels.append(np.mean(y_true))

    maximum_prob = 0
    for idx in range(len(relevant_probs - 8)):
        current = np.mean(relevant_probs[idx:idx + 8])
        if current > maximum_prob:
            maximum_prob = current

    ten_slice_prob.append(maximum_prob)

probabilities = np.asarray(probabilities).reshape(len(probabilities), )
ten_slice_prob = np.asarray(ten_slice_prob).reshape(len(ten_slice_prob), )

true_labels = np.asarray(true_labels).reshape(len(true_labels), )
predictions = np.where(probabilities < 0.5, 0, 1).reshape(len(probabilities), )
ten_slice_prediction = np.where(ten_slice_prob < 0.5, 0, 1).reshape(len(ten_slice_prob), )

results = pd.DataFrame({"Filename": uniqe_filename,
                        "true_labels": true_labels,
                        "Predictions": ten_slice_prediction,
                        "Probabilities": ten_slice_prob})
results.to_csv("results_test_scan-based-1016-end.csv", index=False)

# Accuracy
predictions = ten_slice_prediction
probabilities = ten_slice_prob
acc = accuracy_score(true_labels, predictions)
print('Scan-based Accuracy: ', acc)


# subject based
df = pd.read_csv("results_test_scan-based-1016-end.csv")
file_names = df["Filename"]

# get the subjects and view list
uniqe_filename = []
for name in file_names:
    if name[:8] not in uniqe_filename:
        uniqe_filename.append(name[:8])

probabilities = []
true_labels = []
ten_slice_prob = []
file_names = df["Filename"]
for name in uniqe_filename:
    res = file_names.str.contains(pat=name)
    relevant_probs = df["Probabilities"][res.values]
    y_true = df["true_labels"][res.values]
    # probabilities.append(np.mean(relevant_probs))
    probabilities.append(np.mean(relevant_probs))
    true_labels.append(np.mean(y_true))
probabilities = np.asarray(probabilities).reshape(len(probabilities), )
true_labels = np.asarray(true_labels).reshape(len(true_labels), )
predictions = np.where(probabilities < 0.48, 0, 1).reshape(len(probabilities), )

results = pd.DataFrame({"Filename": uniqe_filename,
                        "true_labels": true_labels,
                        "Predictions": predictions,
                        "Probabilities": probabilities})
results.to_csv("results_test_case-based-1016-end.csv", index=False)


acc = accuracy_score(true_labels, predictions)
print('Case-based Accuracy: ', acc)
