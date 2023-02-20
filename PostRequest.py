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
    DATASET_PATH_TEST = os.path.join(ROOT_DIR, 'Data', 'datasets', 'gym', 'Harry_gym_movements', 'Harry_gym_movements_TEST.ts')

    X_test, y_test = load_from_tsfile(DATASET_PATH_TEST, return_data_type="numpy3D")


    headers = {'Content-type': 'application/json'}

    response = requests.post('http://localhost:5000/predict', data=json.dumps(X_test[0:1].tolist()), headers=headers)
    print(response)


    









if __name__ == "__main__":
    main()