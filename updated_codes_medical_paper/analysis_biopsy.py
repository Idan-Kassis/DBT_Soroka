from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np

results = pd.read_csv('/Users/idankassis/Desktop/Thesis/codes/pythonProject1/predictions/1024-res/results_test_case-based_1024_biopsy.csv')
true_labels = np.asarray(results['true_labels'])
probabilities = np.asarray(results['Probabilities'])
predictions = np.where(probabilities < 0.55, 0, 1).reshape(len(probabilities), )

acc = accuracy_score(true_labels, predictions)
print('Case-based Accuracy: ', acc)

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
print(cm)

# sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sens = tp / (fn + tp)
spec = tn / (tn + fp)
print('Case-based Sensitivity - ', sens)
print('Case-based Specificity - ', spec)

# AUC
auc = roc_auc_score(true_labels, probabilities, average=None)
print('Case-based AUC - ', auc)