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
print(np.shape(y_train))

X_train_acc = X_train.iloc[:,0:3]
X_test_acc = X_test.iloc[:,0:3]
# y_train_acc = y_train.iloc[:,0:3]
# y_test_acc = y_test.iloc[:,0:3]

X_train_gyro = X_train.iloc[:,3:6]
X_test_gyro = X_test.iloc[:,3:6]
# y_train_gyro = y_train.iloc[:,3:6]
# y_test_gyro = y_test.iloc[:,3:6]

def KNN_experiment(X_train, y_train, X_test, y_test, distance_measure):

    clf = KNeighborsTimeSeriesClassifier(distance=distance_measure)


    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()


    predictions = clf.predict(X_test)

    count = 0

    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            count += 1

    accuracy = count/len(y_test)
    train_time = end - start

    return accuracy, train_time


print("********** accel + gyro **********")
print(KNN_experiment(X_train, y_train, X_test, y_test, "euclidean"))
print(KNN_experiment(X_train, y_train, X_test, y_test, "dtw"))

print("********** accel **********")
print(KNN_experiment(X_train_acc, y_train, X_test_acc, y_test, "euclidean"))
print(KNN_experiment(X_train_acc, y_train, X_test_acc, y_test, "dtw"))

print("********** gyro **********")
print(KNN_experiment(X_train_gyro, y_train, X_test_gyro, y_test, "euclidean"))
print(KNN_experiment(X_train_gyro, y_train, X_test_gyro, y_test, "dtw"))


# print(X_train.iloc[:,1:2])

print("********** accel X **********")
print(KNN_experiment(X_train.iloc[:,0:1], y_train, X_test.iloc[:,0:1], y_test, "euclidean"))
print(KNN_experiment(X_train.iloc[:,0:1], y_train, X_test.iloc[:,0:1], y_test, "dtw"))

print("********** accel Y **********")
print(KNN_experiment(X_train.iloc[:,1:2], y_train, X_test.iloc[:,1:2], y_test, "euclidean"))
print(KNN_experiment(X_train.iloc[:,1:2], y_train, X_test.iloc[:,1:2], y_test, "dtw"))

print("********** accel Z **********")
print(KNN_experiment(X_train.iloc[:,2:3], y_train, X_test.iloc[:,2:3], y_test, "euclidean"))
print(KNN_experiment(X_train.iloc[:,2:3], y_train, X_test.iloc[:,2:3], y_test, "dtw"))

print("********** gyro X **********")
print(KNN_experiment(X_train.iloc[:,3:4], y_train, X_test.iloc[:,3:4], y_test, "euclidean"))
print(KNN_experiment(X_train.iloc[:,3:4], y_train, X_test.iloc[:,3:4], y_test, "dtw"))

print("********** gyro Y **********")
print(KNN_experiment(X_train.iloc[:,4:5], y_train, X_test.iloc[:,4:5], y_test, "euclidean"))
print(KNN_experiment(X_train.iloc[:,4:5], y_train, X_test.iloc[:,4:5], y_test, "dtw"))

print("********** gyro Z **********")
print(KNN_experiment(X_train.iloc[:,5:6], y_train, X_test.iloc[:,5:6], y_test, "euclidean"))
print(KNN_experiment(X_train.iloc[:,5:6], y_train, X_test.iloc[:,5:6], y_test, "dtw"))