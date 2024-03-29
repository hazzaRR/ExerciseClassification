"""
Author: Harry Redman


Description: functions that run the time series experiments using a the single train-test split files and saves the results
in a .txt file and a confusion matrix of the photos in the cv_results folder.

"""
import os
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.datasets import load_from_tsfile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import time

def time_series_experiment(X_train, y_train, X_test, y_test, clf, clf_name, filepath, cm_filepath, dataset_name):

    dataset_name = dataset_name.replace('_', ' ')

    dataset_name = dataset_name.split(' ')

    dataset_name = ' '.join(dataset_name[1:])

    """ train classifier and calculate train time """
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()

    """ iterate through the classifiers predictions and compare them to actual class labels to get the accuracy of the classifier """
    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    bal_accuracy = balanced_accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    pre_score = precision_score(y_test, predictions, average='macro')
    r_score = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    auroc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    train_time = end - start

    with open(f'./{filepath}.txt', 'w') as f:
        f.write("Classifier Results\n")
        f.write("------------------------------------------------\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Classifier: {clf_name}\n")
        f.write("------------------------------------------------\n")
        f.write(f"Train time: {train_time}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Balanced Accuracy: {bal_accuracy}\n")
        f.write(f"Precision: {pre_score}\n")
        f.write(f"Recall: {r_score}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"AUROC: {auroc}\n")
        f.write(f"Confusion Matrix:\n {cm}\n")
        f.write("------------------------------------------------\n")


    cmd = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
    # plot and save the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 9))
    cmd.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.xticks(rotation=45)
    plt.title(f"{clf_name} on {dataset_name} train/test split")
    plt.savefig(cm_filepath)




def col_ensemble_experiment(X_train, y_train, X_test, y_test, clf_to_use, clf_name, filepath, cm_filepath, dataset_name):

    rows, cols = np.shape(X_train)

    classifiersToEnsemble = []

    if cols == 1:

        clf = clf_to_use


    else:

        for i in range(cols):
            clfName = "clf" + str(i)
            clf = clf_to_use
            col = [i]

            clfTuple = (clfName, clf, col)
            classifiersToEnsemble.append(clfTuple)

        clf = ColumnEnsembleClassifier(
        estimators=classifiersToEnsemble)
        
        
    time_series_experiment(X_train, y_train, X_test, y_test, clf, clf_name, filepath, cm_filepath, dataset_name)




def main():
    print("hello world")

    CURRENT_PATH = os.getcwd()

    X_train, y_train = load_from_tsfile(
    os.path.join(CURRENT_PATH, 'Data', 'datasets', 'gym', 'Harry_gym_movements',  f"Harry_gym_movements_TRAIN.ts"))
    X_test, y_test = load_from_tsfile(
    os.path.join(CURRENT_PATH, 'Data', 'datasets', 'gym', 'Harry_gym_movements',  f"Harry_gym_movements_TEST.ts"))


    print(np.shape(X_train))

    tsf_clf = TimeSeriesForestClassifier()

    col_ensemble_experiment(X_train, y_train, X_test, y_test, tsf_clf, './', 'harry')

    X_train, y_train = load_from_tsfile(
    os.path.join(CURRENT_PATH, 'Data', 'datasets', 'gym', 'Harry_gym_movements_ax',  f"Harry_gym_movements_ax_TRAIN.ts"))
    X_test, y_test = load_from_tsfile(
    os.path.join(CURRENT_PATH, 'Data', 'datasets', 'gym', 'Harry_gym_movements_Ax',  f"Harry_gym_movements_ax_TEST.ts"))

    col_ensemble_experiment(X_train, y_train, X_test, y_test, tsf_clf, './', 'harry')

if __name__ == "__main__":
    main()
