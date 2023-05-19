"""
Author: Harry Redman


Description: file that loads in all the 50:50 train and test datasets files and runs all the classifiers on the train-test functions made in 
the experiment python file

"""
import os
from experiment import time_series_experiment, col_ensemble_experiment
from sktime.datasets import load_from_tsfile
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np


def run_experiment(clf, clf_name, data_2d_array=False, uni_ts_clf=False):

    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, "Data", "datasets")

    RESULT_PATH = os.path.join(CURRENT_PATH, "results", clf_name)

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        os.makedirs(os.path.join(RESULT_PATH, 'confusion_matrices'))


    for dataset_type in os.listdir(DATA_PATH):

        if dataset_type == "person_independent":
            continue

        CURRENT_DATASET = os.path.join(DATA_PATH, dataset_type)

        for dataset in os.listdir(CURRENT_DATASET):


            if dataset == "person_independent":
                continue


            print(os.path.join(CURRENT_DATASET, dataset, f"{dataset}_TRAIN.ts"))

            """ load in train and test data """

            if data_2d_array:

                X_train, y_train = load_from_tsfile(
                os.path.join(CURRENT_DATASET, dataset, f"{dataset}_TRAIN.ts"), return_data_type="numpy2d"
                )
                X_test, y_test = load_from_tsfile(
                os.path.join(CURRENT_DATASET, dataset, f"{dataset}_TEST.ts"), return_data_type="numpy2d"
                )

            else:
                

                X_train, y_train = load_from_tsfile(
                os.path.join(CURRENT_DATASET, dataset, f"{dataset}_TRAIN.ts")
                )
                X_test, y_test = load_from_tsfile(
                    os.path.join(CURRENT_DATASET, dataset, f"{dataset}_TEST.ts")
                )

            if uni_ts_clf:
                col_ensemble_experiment(X_train, y_train, X_test, y_test, clf, clf_name, f"results/{clf_name}/{dataset}", f"results/{clf_name}/confusion_matrices/{dataset}", dataset)

            else:
                time_series_experiment(X_train, y_train, X_test, y_test, clf, clf_name, f"results/{clf_name}/{dataset}", f"results/{clf_name}/confusion_matrices/{dataset}", dataset)

def main() :

    CURRENT_PATH = os.getcwd()
    RESULT_PATH = os.path.join(CURRENT_PATH, "results")
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    knn_classifier = KNeighborsTimeSeriesClassifier(distance='dtw')
    run_experiment(clf=knn_classifier, clf_name="kNN-DTW")

    rocket_classifier = RocketClassifier(num_kernels=1000)
    run_experiment(clf=rocket_classifier, clf_name="Rocket")

    tsf_classifier = TimeSeriesForestClassifier()
    run_experiment(clf=tsf_classifier, clf_name="Time Series Forest", uni_ts_clf=True)
    
    boss_classifier = BOSSEnsemble()
    run_experiment(clf=boss_classifier, clf_name="Boss", data_2d_array=True)

    dt_classifier = DecisionTreeClassifier()
    run_experiment(clf=dt_classifier,clf_name="Decision Tree", data_2d_array=True)

    nb_classifier = GaussianNB()
    run_experiment(clf=nb_classifier,clf_name="Naive Bayes", data_2d_array=True)

    ada_classifier = AdaBoostClassifier()
    run_experiment(clf=ada_classifier, clf_name="AdaBoost", data_2d_array=True)

    mlp_classifier = MLPClassifier()
    run_experiment(clf=mlp_classifier,clf_name="Multi-layer Perceptron", data_2d_array=True)

    rf_classifier = RandomForestClassifier()
    run_experiment(clf=rf_classifier,clf_name="Random Forest", data_2d_array=True)

    knn_classifier = KNeighborsClassifier()
    run_experiment(clf=knn_classifier, clf_name="kNN-ED", data_2d_array=True)


if __name__ == "__main__":
    main()