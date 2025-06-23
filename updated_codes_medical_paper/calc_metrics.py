import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

results = pd.read_csv("results_test_scan-based-all.csv")
true_labels = list(results["true_labels"])
predictions = list(results["Predictions"])
probabilities = list(results["Probabilities"])

acc = accuracy_score(true_labels, predictions)
print('Scan-based Accuracy: ', acc)

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
print(cm)

# sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sens = tp / (fn + tp)
spec = tn / (tn + fp)
print('Scan-based Sensitivity - ', round(sens, 4))
print('Scan-based Specificity - ', round(spec, 4))

# AUC
auc = roc_auc_score(true_labels, probabilities, average=None)
print('Scan-based AUC - ', round(auc, 4))

fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 4)))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('scan_based_ROC.jpg')
plt.close()
plt.clf()


results = pd.read_csv("results_test_case-based-all.csv")
true_labels = list(results["true_labels"])
predictions = list(results["Predictions"])
probabilities = list(results["Probabilities"])

acc = accuracy_score(true_labels, predictions)
print('Case-based Accuracy: ', acc)

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
print(cm)

# sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sens = tp / (fn + tp)
spec = tn / (tn + fp)
print('Case-based Sensitivity - ', round(sens, 4))
print('Case-based Specificity - ', round(spec, 4))

# AUC
auc = roc_auc_score(true_labels, probabilities, average=None)
print('Case-based AUC - ', round(auc, 4))

fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 4)))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('case_based_ROC.jpg')
plt.close()
plt.clf()
