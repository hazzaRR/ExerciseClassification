import os
from experiment import time_series_experiment
from sktime.datasets import load_from_tsfile
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.hybrid._hivecote_v2 import HIVECOTEV2
from sktime.classification.kernel_based import RocketClassifier


def main():
    print("hello world")

    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, "Data", "datasets")
    

    """ load in train and test data """
    X_train, y_train = load_from_tsfile(
        os.path.join(DATA_PATH, "gym/Harry_gym_movements/Harry_gym_movements_TRAIN.ts")
    )
    X_test, y_test = load_from_tsfile(
        os.path.join(DATA_PATH, "gym/Harry_gym_movements/Harry_gym_movements_TEST.ts")
    )

    knn_classifier = KNeighborsTimeSeriesClassifier(distance='dtw')
    rocket_clf = RocketClassifier(num_kernels=1000)
    hc2_clf = HIVECOTEV2()


    print(time_series_experiment(X_train, y_train, X_test, y_test, hc2_clf))

if __name__ == "__main__":
    main()