import os
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_from_tsfile
import numpy as np
import time

def KNN_experiment(X_train, y_train, X_test, y_test, distance_measure):

    clf = KNeighborsTimeSeriesClassifier(distance=distance_measure)

    """ train classifier and calculate train time """
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()

    """ iterate through the classifiers predictions and compare them to actual class labels to get the accuracy of the classifier """
    predictions = clf.predict(X_test)

    count = 0

    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            count += 1

    accuracy = count/len(y_test)
    train_time = end - start

    return accuracy, train_time


def main():
    print("hello world")

if __name__ == "__main__":
    main()