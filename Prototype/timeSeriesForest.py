import os
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.datasets import load_from_tsfile
import numpy as np
import time

def tsf_experiment_univariate(X_train_data, y_train_data, X_test_data, y_test_data, n_estimators_value):

    """ builds a time series forest classifiers off of a single time series axis that it is passed in the parameters"""
    clf = TimeSeriesForestClassifier(n_estimators=n_estimators_value)

    """ train classifier and calculate train time """
    start = time.time()
    clf.fit(X_train_data, y_train_data)
    end = time.time()


    """ iterate through the classifiers predictions and compare them to actual class labels to get the accuracy of the classifier """
    predictions = clf.predict(X_test_data)

    count = 0

    for i in range(len(y_test_data)):
        if predictions[i] == y_test_data[i]:
            count += 1

    accuracy = count/len(y_test_data)
    train_time = end - start

    return accuracy, train_time

def tsf_experiment_mulitvariate(X_train_data, y_train_data, X_test_data, y_test_data, n_estimators_value):

    """
    For multivariate time series forest I use Column Ensembler to get the result from each time series axis
    and aggregate them together to get an overall prediction for the data instance
    """

    rows, cols = np.shape(X_train_data)

    classifiersToEnsemble = []

    for i in range(cols):
        clfName = "TSF" + str(i)
        clf = TimeSeriesForestClassifier(n_estimators=n_estimators_value)
        col = [i]

        clfTuple = (clfName, clf, col)
        classifiersToEnsemble.append(clfTuple)

    clf = ColumnEnsembleClassifier(
    estimators=classifiersToEnsemble)

    start = time.time()
    clf.fit(X_train_data, y_train_data)
    end = time.time()


    predictions = clf.predict(X_test_data)

    count = 0

    for i in range(len(y_test_data)):
        if predictions[i] == y_test_data[i]:
            count += 1

    accuracy = count/len(y_test_data)
    train_time = end - start

    return accuracy, train_time


def main():
    print("")
if __name__ == "__main__":
    main()