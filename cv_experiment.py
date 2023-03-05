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
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
import time
from sklearn.preprocessing import LabelEncoder


def main() :

    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, "Data", "datasets")

    X, y =  load_from_tsfile(
            os.path.join(DATA_PATH, "gym", "Harry_gym_movements", "Harry_gym_movements.ts"), return_data_type="numpy2d")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)


    print(y_encoded)
    print(le.inverse_transform(y_encoded))

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # model = KNeighborsClassifier()
    model = RocketClassifier(num_kernels=1000)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'bal_accuracy': make_scorer(balanced_accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        # 'auroc': make_scorer(roc_auc_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro'),
        # 'confusion_matrix': make_scorer(confusion_matrix)
    }

    results = cross_validate(model, X, y_encoded, cv=kfold, scoring=scoring, return_train_score=True)

    print(results)

    # # print the results
    # for metric, values in results.items():
    #     if metric.endswith('_time'):
    #         print(f'{metric}: {values.mean()} +/- {values.std()} seconds')
    #     elif metric.endswith('_confusion_matrix'):
    #         print(f'{metric}: {values}')
    #     else:
    #         print(f'{metric}: {values.mean()} +/- {values.std()}')


def test():

    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, "Data", "datasets")

    # load an example time series dataset
    X, y =  load_from_tsfile(
            os.path.join(DATA_PATH, "gym", "Harry_gym_movements", "Harry_gym_movements.ts"), return_data_type="numpy2d")
    
    le = LabelEncoder()
    y = le.fit_transform(y)


    # assume X is your feature matrix and y is your target variable
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    print(kfold)

    # model = KNeighborsClassifier()
    model = RocketClassifier(num_kernels=1000)


    y_pred = cross_val_predict(model, X, y, cv=kfold)


        # evaluate performance metrics
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    # roc_auc = roc_auc_score(y, y_pred, average='weighted', multi_class='ovr')
    f1 = f1_score(y, y_pred, average='weighted')
    confusion = confusion_matrix(y, y_pred)

    # print results
    print("Accuracy:", accuracy)
    print("Balanced accuracy:", balanced_accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    # print("AUROC:", roc_auc)
    print("F1 score:", f1)
    print("Confusion matrix:\n", confusion)






if __name__ == "__main__":
    # main()
    test()