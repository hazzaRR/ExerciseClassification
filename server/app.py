from flask import Flask, request
import joblib
import json
import os
import numpy as np

app = Flask(__name__)

ROOT_DIR = os.getcwd()


def preprocess_data(data):

    '''
    normalise the sensor data recieved from the request and split it into 3 seperate data instances to predict on
    '''

    data = np.array(data)
    normalised_data = []


    for axis in data:
        axis = axis[50:150]
        # axis = (axis - np.mean(axis)) / np.std(axis)
        axis = (axis-axis.min())/(axis.max()-axis.min())
        normalised_data.append(axis)


    normalised_data = np.array([normalised_data])
    print(normalised_data)

    return normalised_data


rocket_classifier = joblib.load(os.path.join(ROOT_DIR, 'rocket_model.joblib'))


@app.route('/predict', methods=['POST'])

def predict():

    data = request.get_json()

    sensorData = preprocess_data(data)

    print(sensorData)

    prediction = rocket_classifier.predict(sensorData)

    print(prediction)

    return list(prediction)



@app.route('/test_predict', methods=['POST'])

def test_predict():

    data = request.get_json()

    data = np.array(data)

    prediction = rocket_classifier.predict(data)

    print(prediction)

    return prediction[0]



if __name__ == '__main__':
    app.run()

