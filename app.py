from flask import Flask, request
import joblib
import json
import os
import numpy as np

app = Flask(__name__)

ROOT_DIR = os.getcwd()


rocket_classifier = joblib.load(os.path.join(ROOT_DIR, 'rocket_model.joblib'))


@app.route('/predict', methods=['POST'])

def predict():

    data = json.loads(request.data.decode('utf-8'))

    # print(np.shape(data['values']))
    # print(data['values'])

    data = np.array(data)
    print(data)

    prediction = rocket_classifier.predict(data)

    print(prediction)

    return prediction[0]



if __name__ == '__main__':
    app.run(debug=True)

