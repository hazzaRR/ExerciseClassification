import os
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.datasets import load_from_tsfile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import numpy as np
import time

def time_series_experiment(X_train, y_train, X_test, y_test, clf):

    """ train classifier and calculate train time """
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()

    """ iterate through the classifiers predictions and compare them to actual class labels to get the accuracy of the classifier """
    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    bal_accuracy = balanced_accuracy_score(y_test, predictions)
    confusion_matrix(y_test, predictions)
    train_time = end - start

    return accuracy, train_time, confusion_matrix, bal_accuracy


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
