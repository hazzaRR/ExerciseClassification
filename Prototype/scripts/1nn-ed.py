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


def save_results(filename, accel_gyro_acc, accel_gyro_time, accel_acc, accel_time, gyro_acc, gyro_time, \
    accel_x_acc, accel_x_time, accel_y_acc, accel_y_time, accel_z_acc, accel_z_time, gyro_x_acc, gyro_x_time,\
        gyro_y_acc, gyro_y_time, gyro_z_acc, gyro_z_time):

    with open(f'./Prototype/output/{filename}.txt', 'w') as f:
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

def print_results(accel_gyro_acc, accel_gyro_time, accel_acc, accel_time, gyro_acc, gyro_time, \
    accel_x_acc, accel_x_time, accel_y_acc, accel_y_time, accel_z_acc, accel_z_time, gyro_x_acc, gyro_x_time,\
        gyro_y_acc, gyro_y_time, gyro_z_acc, gyro_z_time):

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

    
""" get just the accelerometer data columns"""
X_train_acc = X_train.iloc[:,0:3]
X_test_acc = X_test.iloc[:,0:3]

""" get just the gyroscope data columns """
X_train_gyro = X_train.iloc[:,3:6]
X_test_gyro = X_test.iloc[:,3:6]


"""K-NN with Dynamic time warping distance measure"""
ed_accel_gyro_acc, ed_accel_gyro_time = KNN_experiment(X_train, y_train, X_test, y_test, "euclidean")
ed_accel_acc, ed_accel_time = KNN_experiment(X_train_acc, y_train, X_test_acc, y_test, "euclidean")
ed_gyro_acc, ed_gyro_time = KNN_experiment(X_train_gyro, y_train, X_test_gyro, y_test, "euclidean")
ed_accel_x_acc, ed_accel_x_time = KNN_experiment(X_train.iloc[:,0:1], y_train, X_test.iloc[:,0:1], y_test, "euclidean")
ed_accel_y_acc, ed_accel_y_time = KNN_experiment(X_train.iloc[:,1:2], y_train, X_test.iloc[:,1:2], y_test, "euclidean")
ed_accel_z_acc, ed_accel_z_time = KNN_experiment(X_train.iloc[:,2:3], y_train, X_test.iloc[:,2:3], y_test, "euclidean")
ed_gyro_x_acc, ed_gyro_x_time = KNN_experiment(X_train.iloc[:,3:4], y_train, X_test.iloc[:,3:4], y_test, "euclidean")
ed_gyro_y_acc, ed_gyro_y_time = KNN_experiment(X_train.iloc[:,4:5], y_train, X_test.iloc[:,4:5], y_test, "euclidean")
ed_gyro_z_acc, ed_gyro_z_time = KNN_experiment(X_train.iloc[:,5:6], y_train, X_test.iloc[:,5:6], y_test, "euclidean")



"""K-NN with Dynamic time warping distance measure"""
dtw_accel_gyro_acc, dtw_accel_gyro_time = KNN_experiment(X_train, y_train, X_test, y_test, "dtw")
dtw_accel_acc, dtw_accel_time = KNN_experiment(X_train_acc, y_train, X_test_acc, y_test, "dtw")
dtw_gyro_acc, dtw_gyro_time = KNN_experiment(X_train_gyro, y_train, X_test_gyro, y_test, "dtw")
dtw_accel_x_acc, dtw_accel_x_time = KNN_experiment(X_train.iloc[:,0:1], y_train, X_test.iloc[:,0:1], y_test, "dtw")
dtw_accel_y_acc, dtw_accel_y_time = KNN_experiment(X_train.iloc[:,1:2], y_train, X_test.iloc[:,1:2], y_test, "dtw")
dtw_accel_z_acc, dtw_accel_z_time = KNN_experiment(X_train.iloc[:,2:3], y_train, X_test.iloc[:,2:3], y_test, "dtw")
dtw_gyro_x_acc, dtw_gyro_x_time = KNN_experiment(X_train.iloc[:,3:4], y_train, X_test.iloc[:,3:4], y_test, "dtw")
dtw_gyro_y_acc, dtw_gyro_y_time = KNN_experiment(X_train.iloc[:,4:5], y_train, X_test.iloc[:,4:5], y_test, "dtw")
dtw_gyro_z_acc, dtw_gyro_z_time = KNN_experiment(X_train.iloc[:,5:6], y_train, X_test.iloc[:,5:6], y_test, "dtw")


save_results("ed_knn_experiment_results", ed_accel_gyro_acc, ed_accel_gyro_time, ed_accel_acc, ed_accel_time, ed_gyro_acc, ed_gyro_time, \
    ed_accel_x_acc, ed_accel_x_time, ed_accel_y_acc, ed_accel_y_time, ed_accel_z_acc, ed_accel_z_time, ed_gyro_x_acc, ed_gyro_x_time,\
        ed_gyro_y_acc, ed_gyro_y_time, ed_gyro_z_acc, ed_gyro_z_time)

print_results(ed_accel_gyro_acc, ed_accel_gyro_time, ed_accel_acc, ed_accel_time, ed_gyro_acc, ed_gyro_time, \
    ed_accel_x_acc, ed_accel_x_time, ed_accel_y_acc, ed_accel_y_time, ed_accel_z_acc, ed_accel_z_time, ed_gyro_x_acc, ed_gyro_x_time,\
        ed_gyro_y_acc, ed_gyro_y_time, ed_gyro_z_acc, ed_gyro_z_time)

save_results("dtw_knn_experiment_results", dtw_accel_gyro_acc, dtw_accel_gyro_time, dtw_accel_acc, dtw_accel_time, dtw_gyro_acc, dtw_gyro_time, \
    dtw_accel_x_acc, dtw_accel_x_time, dtw_accel_y_acc, dtw_accel_y_time, dtw_accel_z_acc, dtw_accel_z_time, dtw_gyro_x_acc, dtw_gyro_x_time,\
        dtw_gyro_y_acc, dtw_gyro_y_time, dtw_gyro_z_acc, dtw_gyro_z_time)

print_results(dtw_accel_gyro_acc, dtw_accel_gyro_time, dtw_accel_acc, dtw_accel_time, dtw_gyro_acc, dtw_gyro_time, \
    dtw_accel_x_acc, dtw_accel_x_time, dtw_accel_y_acc, dtw_accel_y_time, dtw_accel_z_acc, dtw_accel_z_time, dtw_gyro_x_acc, dtw_gyro_x_time,\
        dtw_gyro_y_acc, dtw_gyro_y_time, dtw_gyro_z_acc, dtw_gyro_z_time)