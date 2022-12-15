import os
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sktime.datasets import load_from_tsfile, load_from_tsfile_to_dataframe
import numpy as np
import time
import pandas as pd

def sk_experiment(X_train, y_train, X_test, y_test, clf):

    """ train classifier and calculate train time """
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()

    """ iterate through the classifiers predictions and compare them to actual class labels to get the accuracy of the classifier """
    predictions = clf.predict(X_test)

    count = 0

    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            count += 1

    accuracy = count/len(y_test)
    train_time = end - start

    return accuracy, train_time


def main():

    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, "Prototype", "Prototype_2", "data")
    
    X_train, y_train = load_from_tsfile(
    os.path.join(DATA_PATH, "Powerlift_movements/Powerlift_movements_TRAIN.ts"), return_data_type="numpy2d"
    )
    X_test, y_test = load_from_tsfile(
        os.path.join(DATA_PATH, "Powerlift_movements/Powerlift_movements_TEST.ts"), return_data_type="numpy2d"
    )



    print("****")
    hey = X_train
    df = pd.DataFrame(hey)

    print(np.shape(X_train))
    # print(df)

    svc_clf = SVC()
    dt_clf = DecisionTreeClassifier(random_state=1);
    nb_clf = GaussianNB()

    accel_x_acc, accel_x_time = sk_experiment(X_train[:,0:100], y_train, X_test[:,0:100], y_test, dt_clf)
    print(accel_x_acc, accel_x_time)

    accel_x_acc, accel_x_time = sk_experiment(X_train, y_train, X_test, y_test, svc_clf)
    print(accel_x_acc, accel_x_time)
    accel_x_acc, accel_x_time = sk_experiment(X_train, y_train, X_test, y_test, dt_clf)
    print(accel_x_acc, accel_x_time)
    accel_x_acc, accel_x_time = sk_experiment(X_train, y_train, X_test, y_test, nb_clf)
    print(accel_x_acc, accel_x_time)

if __name__ == "__main__":
    main()