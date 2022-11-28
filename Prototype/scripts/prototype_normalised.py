from timeSeriesForest import tsf_experiment_mulitvariate, tsf_experiment_univariate
from knn import KNN_experiment, save_results, print_results
from output_functions import save_results, print_results
from sktime.datasets import load_from_tsfile
import os

def main():
    
    CURRENT_PATH = os.getcwd()

    DATA_PATH = os.path.join(CURRENT_PATH, "Prototype", "Prototype_2", "data")

    X_train, y_train = load_from_tsfile(
        os.path.join(DATA_PATH, "Powerlift_movements/Powerlift_movements_TRAIN.ts")
    )
    X_test, y_test = load_from_tsfile(
        os.path.join(DATA_PATH, "Powerlift_movements/Powerlift_movements_TEST.ts")
    )
        
    """ get just the accelerometer data columns"""
    X_train_acc = X_train.iloc[:,0:3]
    X_test_acc = X_test.iloc[:,0:3]

    """ get just the gyroscope data columns """
    X_train_gyro = X_train.iloc[:,3:6]
    X_test_gyro = X_test.iloc[:,3:6]


    """K-NN with euclidean distance measure"""
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

    """Time series forest results"""
    tsf_accel_gyro_acc, tsf_accel_gyro_time = tsf_experiment_mulitvariate(X_train, y_train, X_test, y_test, 5)
    tsf_accel_acc, tsf_accel_time = tsf_experiment_mulitvariate(X_train_acc, y_train, X_test_acc, y_test, 5)
    tsf_gyro_acc, tsf_gyro_time = tsf_experiment_mulitvariate(X_train_gyro, y_train, X_test_gyro, y_test, 5)
    tsf_accel_x_acc, tsf_accel_x_time = tsf_experiment_univariate(X_train.iloc[:,0:1], y_train, X_test.iloc[:,0:1], y_test, 5)
    tsf_accel_y_acc, tsf_accel_y_time = tsf_experiment_univariate(X_train.iloc[:,1:2], y_train, X_test.iloc[:,1:2], y_test, 5)
    tsf_accel_z_acc, tsf_accel_z_time = tsf_experiment_univariate(X_train.iloc[:,2:3], y_train, X_test.iloc[:,2:3], y_test, 5)
    tsf_gyro_x_acc, tsf_gyro_x_time = tsf_experiment_univariate(X_train.iloc[:,3:4], y_train, X_test.iloc[:,3:4], y_test, 5)
    tsf_gyro_y_acc, tsf_gyro_y_time = tsf_experiment_univariate(X_train.iloc[:,4:5], y_train, X_test.iloc[:,4:5], y_test, 5)
    tsf_gyro_z_acc, tsf_gyro_z_time = tsf_experiment_univariate(X_train.iloc[:,5:6], y_train, X_test.iloc[:,5:6], y_test, 5)


    save_results("Prototype/Prototype_2/output/ed_knn_experiment_results", ed_accel_gyro_acc, ed_accel_gyro_time, ed_accel_acc, ed_accel_time, ed_gyro_acc, ed_gyro_time, \
        ed_accel_x_acc, ed_accel_x_time, ed_accel_y_acc, ed_accel_y_time, ed_accel_z_acc, ed_accel_z_time, ed_gyro_x_acc, ed_gyro_x_time,\
            ed_gyro_y_acc, ed_gyro_y_time, ed_gyro_z_acc, ed_gyro_z_time)

    print("********************")
    print("Euclidean Distance K-NN experiment")

    print_results(ed_accel_gyro_acc, ed_accel_gyro_time, ed_accel_acc, ed_accel_time, ed_gyro_acc, ed_gyro_time, \
        ed_accel_x_acc, ed_accel_x_time, ed_accel_y_acc, ed_accel_y_time, ed_accel_z_acc, ed_accel_z_time, ed_gyro_x_acc, ed_gyro_x_time,\
            ed_gyro_y_acc, ed_gyro_y_time, ed_gyro_z_acc, ed_gyro_z_time)

    save_results("Prototype/Prototype_2/output/dtw_knn_experiment_results", dtw_accel_gyro_acc, dtw_accel_gyro_time, dtw_accel_acc, dtw_accel_time, dtw_gyro_acc, dtw_gyro_time, \
        dtw_accel_x_acc, dtw_accel_x_time, dtw_accel_y_acc, dtw_accel_y_time, dtw_accel_z_acc, dtw_accel_z_time, dtw_gyro_x_acc, dtw_gyro_x_time,\
            dtw_gyro_y_acc, dtw_gyro_y_time, dtw_gyro_z_acc, dtw_gyro_z_time)

    print("********************")
    print("Dynamic Time Warping K-NN experiment")

    print_results(dtw_accel_gyro_acc, dtw_accel_gyro_time, dtw_accel_acc, dtw_accel_time, dtw_gyro_acc, dtw_gyro_time, \
        dtw_accel_x_acc, dtw_accel_x_time, dtw_accel_y_acc, dtw_accel_y_time, dtw_accel_z_acc, dtw_accel_z_time, dtw_gyro_x_acc, dtw_gyro_x_time,\
            dtw_gyro_y_acc, dtw_gyro_y_time, dtw_gyro_z_acc, dtw_gyro_z_time)

    print("********************")
    print("Time Series Forest experiment")

    save_results("Prototype/Prototype_2/output/tsf_experiment_results", tsf_accel_gyro_acc, tsf_accel_gyro_time, tsf_accel_acc, tsf_accel_time, tsf_gyro_acc, tsf_gyro_time, \
        tsf_accel_x_acc, tsf_accel_x_time, tsf_accel_y_acc, tsf_accel_y_time, tsf_accel_z_acc, tsf_accel_z_time, tsf_gyro_x_acc, tsf_gyro_x_time,\
            tsf_gyro_y_acc, tsf_gyro_y_time, tsf_gyro_z_acc, tsf_gyro_z_time)

    print_results( tsf_accel_gyro_acc, tsf_accel_gyro_time, tsf_accel_acc, tsf_accel_time, tsf_gyro_acc, tsf_gyro_time, \
        tsf_accel_x_acc, tsf_accel_x_time, tsf_accel_y_acc, tsf_accel_y_time, tsf_accel_z_acc, tsf_accel_z_time, tsf_gyro_x_acc, tsf_gyro_x_time,\
            tsf_gyro_y_acc, tsf_gyro_y_time, tsf_gyro_z_acc, tsf_gyro_z_time)

if __name__ == "__main__":
    main()