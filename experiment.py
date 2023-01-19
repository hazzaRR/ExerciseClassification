import os
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.datasets import load_from_tsfile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import time

def time_series_experiment(X_train, y_train, X_test, y_test, clf, filepath, dataset_name):

    """ train classifier and calculate train time """
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()

    """ iterate through the classifiers predictions and compare them to actual class labels to get the accuracy of the classifier """
    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    bal_accuracy = balanced_accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    pre_score = precision_score(y_test, predictions, average='macro')
    r_score = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    auroc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    print(cm)
    train_time = end - start

    with open(f'./{filepath}.txt', 'w') as f:
        f.write("Classifier Results\n")
        f.write("------------------------------------------------\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Classifier: {str(clf)}\n")
        f.write("------------------------------------------------\n")
        f.write(f"Train time: {train_time}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Balanced Accuracy: {bal_accuracy}\n")
        f.write(f"Precision: {pre_score}\n")
        f.write(f"Recall: {r_score}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"AUROC: {auroc}\n")
        f.write(f"Confusion Matrix:\n {cm}\n")
        f.write("------------------------------------------------\n")



    return accuracy, train_time, cm, bal_accuracy


def col_ensemble_experiment(X_train, y_train, X_test, y_test, clf_to_use):

    rows, cols = np.shape(X_train)

    classifiersToEnsemble = []

    for i in range(cols):
        clfName = "clf" + str(i)
        clf = clf_to_use
        col = [i]

        clfTuple = (clfName, clf, col)
        classifiersToEnsemble.append(clfTuple)

    clf = ColumnEnsembleClassifier(
    estimators=classifiersToEnsemble)


    accuracy, train_time, confusion_matrix, bal_accuracy = time_series_experiment(X_train, y_train, X_test, y_test, clf)


    return accuracy, train_time, confusion_matrix, bal_accuracy


def main():
    print("hello world")

if __name__ == "__main__":
    main()
