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

    DATASET_PATH = os.path.join(ROOT_DIR, 'Data', 'datasets', 'gym', 'Harry_gym_movements', 'Harry_gym_movements_TRAIN.ts')
    DATASET_PATH_TEST = os.path.join(ROOT_DIR, 'Data', 'datasets', 'gym', 'Harry_gym_movements', 'Harry_gym_movements_TEST.ts')

    rocket_classifier = RocketClassifier(num_kernels=1000)


    X_train, y_train = load_from_tsfile(DATASET_PATH)
    X_test, y_test = load_from_tsfile(DATASET_PATH)

    rocket_classifier.fit(X_train, y_train)

    # joblib.dump(rocket_classifier, 'rocket_model.joblib')

    predicitions = rocket_classifier.predict(X_test.iloc[0:1])

    print(predicitions)


    """ trying to send numpy array as json object """
    print(np.shape(X_test))

    data = X_test.iloc[0:1].to_json(orient ='columns')


    df = pd.read_json(data, orient ='columns')

    print(data.info())
    # predicitions = rocket_classifier.predict(df)

    # print(predicitions)


    # data = data.to_frame()

    # print(df.head(1))


    # headers = {'Content-type': 'application/json'}

    # response = requests.post('http://localhost:5000/predict', data=json.dumps(data), headers=headers)


    









if __name__ == "__main__":
    main()
