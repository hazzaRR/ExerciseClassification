import os
from cv_experiment import time_series_experiment, col_ensemble_experiment
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


def run_experiment(clf, sklearn_clf=False, uni_ts_clf=False):

    clf_name = str(clf).split('(')[0]

    # print(clf_name)
    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, "Data", "datasets")

    RESULT_PATH = os.path.join(CURRENT_PATH, "cv_results", clf_name)

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)


    for dataset_type in os.listdir(DATA_PATH):

        CURRENT_DATASET = os.path.join(DATA_PATH, dataset_type)

        for dataset in os.listdir(CURRENT_DATASET):

            print(os.path.join(CURRENT_DATASET, dataset, f"{dataset}.ts"))

            """ load in train and test data """

            if sklearn_clf:

                X, y = load_from_tsfile(
                os.path.join(CURRENT_DATASET, dataset, f"{dataset}.ts"), return_data_type="numpy2d"
                )

            else:
                
                X, y = load_from_tsfile(
                os.path.join(CURRENT_DATASET, dataset, f"{dataset}.ts"), return_data_type="numpy3d"
                )


            if uni_ts_clf:
                print(col_ensemble_experiment(X, y, clf, f"cv_results/{clf_name}/{dataset}_cv", dataset))

            else:
            # print(X_train)
                print(time_series_experiment(X, y, clf, f"cv_results/{clf_name}/{dataset}_cv", dataset))

def main() :

    CURRENT_PATH = os.getcwd()
    RESULT_PATH = os.path.join(CURRENT_PATH, "results")
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    # knn_classifier = KNeighborsTimeSeriesClassifier(distance='dtw')
    # run_experiment(clf=knn_classifier)

    # rocket_classifier = RocketClassifier(num_kernels=1000)
    # run_experiment(clf=rocket_classifier)

    tsf_classifier = TimeSeriesForestClassifier()
    run_experiment(clf=tsf_classifier, uni_ts_clf=True)
    
    # boss_classifier = BOSSEnsemble()
    # run_experiment(clf=boss_classifier, uni_ts_clf=True)

    # dt_classifier = DecisionTreeClassifier()
    # run_experiment(clf=dt_classifier, sklearn_clf=True)

    # nb_classifier = GaussianNB()
    # run_experiment(clf=nb_classifier, sklearn_clf=True)

    # ada_classifier = AdaBoostClassifier()
    # run_experiment(clf=ada_classifier, sklearn_clf=True)

    # mlp_classifier = MLPClassifier()
    # run_experiment(clf=mlp_classifier, sklearn_clf=True)

    # rf_classifier = RandomForestClassifier()
    # run_experiment(clf=rf_classifier, sklearn_clf=True)

    # knn_classifier = KNeighborsClassifier()
    # run_experiment(clf=knn_classifier, sklearn_clf=True)


if __name__ == "__main__":
    main()