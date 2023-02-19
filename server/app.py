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

    instance1 = []
    instance2 = []
    instance3 = []

    for axis in data:
        axis = (axis - np.mean(axis)) / np.std(axis)

        instance1.append(axis[000:100])
        instance2.append(axis[100:200])
        instance3.append(axis[200:300])


    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    instance3 = np.array(instance3)


    data = np.array([instance1, instance2, instance3])

    return data


rocket_classifier = joblib.load(os.path.join(ROOT_DIR, 'rocket_model.joblib'))


@app.route('/predict', methods=['POST'])

def predict():

    data = request.get_json()

    data = preprocess_data(data)

    prediction = rocket_classifier.predict(data)

    print(prediction)

    return list(prediction)



if __name__ == '__main__':
    app.run(debug=True)

