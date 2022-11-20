import os
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_from_tsfile
import numpy as np
import time

CURRENT_PATH = os.getcwd()

DATA_PATH = os.path.join(CURRENT_PATH, "Prototype", "data")

X_train, y_train = load_from_tsfile(
    os.path.join(DATA_PATH, "Powerlift_movements/Powerlift_movements_TRAIN.ts")
)
X_test, y_test = load_from_tsfile(
    os.path.join(DATA_PATH, "Powerlift_movements/Powerlift_movements_TEST.ts")
)


print(np.shape(X_train.iloc[:,0:3]))

X_train = X_train.iloc[:,3:6]
X_test = X_test.iloc[:,3:6]


def KNN_experiment(sensor_data, distance_measure):

    clf = KNeighborsTimeSeriesClassifier(distance="euclidean")


    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()


    predictions = clf.predict(X_test)

    count = 0

    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            count += 1

    print(count/len(y_test))
    print(end - start)

