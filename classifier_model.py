import os
from sktime.datasets import load_from_tsfile
from sktime.classification.kernel_based import RocketClassifier
import numpy as np
import pandas as pd
import joblib
import requests
import json


def main():

    ROOT_DIR = os.getcwd()

    print(ROOT_DIR)

    DATASET_PATH = os.path.join(ROOT_DIR, 'Data', 'datasets', 'gym', 'Harry_gym_movements', 'Harry_gym_movements_TRAIN.ts',)
    DATASET_PATH_TEST = os.path.join(ROOT_DIR, 'Data', 'datasets', 'gym', 'Harry_gym_movements', 'Harry_gym_movements_TEST.ts')

    rocket_classifier = RocketClassifier(num_kernels=1000)


    X_train, y_train = load_from_tsfile(DATASET_PATH, return_data_type="numpy3D")
    X_test, y_test = load_from_tsfile(DATASET_PATH_TEST, return_data_type="numpy3D")


    print(np.shape(X_test[0:1]))

    rocket_classifier.fit(X_train, y_train)

    joblib.dump(rocket_classifier, './server/rocket_model.joblib')



    









if __name__ == "__main__":
    main()
