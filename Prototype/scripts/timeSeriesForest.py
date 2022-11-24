import os
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.datasets import load_from_tsfile
import numpy as np
import time

CURRENT_PATH = os.getcwd()


"""
load in time series data from ts files
"""

DATA_PATH = os.path.join(CURRENT_PATH, "Prototype", "data")

X_train, y_train = load_from_tsfile(
    os.path.join(DATA_PATH, "Powerlift_movements/Powerlift_movements_TRAIN.ts")
)
X_test, y_test = load_from_tsfile(
    os.path.join(DATA_PATH, "Powerlift_movements/Powerlift_movements_TEST.ts")
)

def tsf_experiment_univariate(X_train_data, y_train_data, X_test_data, y_test_data, n_estimators_value):

    """ builds a time series forest classifiers off of a single time series axis that it is passed in the parameters"""
    clf = TimeSeriesForestClassifier(n_estimators=n_estimators_value)


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


""" get just the accelerometer data columns"""
X_train_acc = X_train.iloc[:,0:3]
X_test_acc = X_test.iloc[:,0:3]

""" get just the gyroscope data columns """
X_train_gyro = X_train.iloc[:,3:6]
X_test_gyro = X_test.iloc[:,3:6]

accel_gyro_acc, accel_gyro_time = tsf_experiment_mulitvariate(X_train, y_train, X_test, y_test, 5)
accel_acc, accel_time = tsf_experiment_mulitvariate(X_train_acc, y_train, X_test_acc, y_test, 5)
gyro_acc, gyro_time = tsf_experiment_mulitvariate(X_train_gyro, y_train, X_test_gyro, y_test, 5)
accel_x_acc, accel_x_time = tsf_experiment_univariate(X_train.iloc[:,0:1], y_train, X_test.iloc[:,0:1], y_test, 5)
accel_y_acc, accel_y_time = tsf_experiment_univariate(X_train.iloc[:,1:2], y_train, X_test.iloc[:,1:2], y_test, 5)
accel_z_acc, accel_z_time = tsf_experiment_univariate(X_train.iloc[:,2:3], y_train, X_test.iloc[:,2:3], y_test, 5)
gyro_x_acc, gyro_x_time = tsf_experiment_univariate(X_train.iloc[:,3:4], y_train, X_test.iloc[:,3:4], y_test, 5)
gyro_y_acc, gyro_y_time = tsf_experiment_univariate(X_train.iloc[:,4:5], y_train, X_test.iloc[:,4:5], y_test, 5)
gyro_z_acc, gyro_z_time = tsf_experiment_univariate(X_train.iloc[:,5:6], y_train, X_test.iloc[:,5:6], y_test, 5)

with open('./Prototype/output/tsf_experiment_results.txt', 'w') as f:
    f.write("Multivariate Results:\n")
    f.write(f" - Accelerometer and Gyroscope Accuracy = {accel_gyro_acc} and train time = {accel_gyro_time}\n")
    f.write(f" - Accelerometer Accuracy = {accel_acc} and train time = {accel_time}\n")
    f.write(f" - Gyroscope Accuracy = {gyro_acc} and train time = {gyro_time}\n")
    f.write("Univariate Results:\n")
    f.write(f" - Accelerometer X axis = {accel_x_acc} and train time = {accel_x_time}\n")
    f.write(f" - Accelerometer Y axis = {accel_y_acc} and train time = {accel_y_time}\n")
    f.write(f" - Accelerometer Z axis = {accel_z_acc} and train time = {accel_z_time}\n")
    f.write(f" - Gyroscope X axis Accuracy = {gyro_x_acc} and train time = {gyro_x_time}\n")
    f.write(f" - Gyroscope Y axis Accuracy = {gyro_y_acc} and train time = {gyro_y_time}\n")
    f.write(f" - Gyroscope Z axis Accuracy = {gyro_z_acc} and train time = {gyro_z_time}\n")


print("Multivariate Results:")
print(f" - Accelerometer and Gyroscope Accuracy = {accel_gyro_acc} and train time = {accel_gyro_time}")
print(f" - Accelerometer Accuracy = {accel_acc} and train time = {accel_time}")
print(f" - Gyroscope Accuracy = {gyro_acc} and train time = {gyro_time}")
print("Univariate Results:")
print(f" - Accelerometer X axis = {accel_x_acc} and train time = {accel_x_time}")
print(f" - Accelerometer Y axis = {accel_y_acc} and train time = {accel_y_time}")
print(f" - Accelerometer Z axis = {accel_z_acc} and train time = {accel_z_time}")
print(f" - Gyroscope X axis Accuracy = {gyro_x_acc} and train time = {gyro_x_time}")
print(f" - Gyroscope Y axis Accuracy = {gyro_y_acc} and train time = {gyro_y_time}")
print(f" - Gyroscope Z axis Accuracy = {gyro_z_acc} and train time = {gyro_z_time}")
