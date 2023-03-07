import os
from sktime.datasets import load_from_tsfile
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import time
from sklearn.metrics import confusion_matrix
from sktime.classification.compose import ColumnEnsembleClassifier


def time_series_experiment(X, y, clf, filepath, dataset_name):

    # initialise empty lists to store results
    train_times = []
    acc_scores = []
    bal_acc_scores = []
    prec_scores = []
    recall_scores = []
    f1_scores = []
    auroc_scores = []
    # initialise lists to store true and predicted labels, to create final confusion matrix
    y_true_all = []
    y_pred_all = []

    """ create a 10 fold cross validation """
    cv = KFold(n_splits=10, shuffle=True, random_state=44)

    # loop over the splits and fit the classifier for each one
    for train_idx, test_idx in cv.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]  # slice X along the first axis
        y_train, y_test = y[train_idx], y[test_idx]  # slice y
        # X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # slice X along the first axis
        # y_train, y_test = y[train_idx], y[test_idx]  # slice y

        """ get time to fit the classifier with the training data """
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()


        y_pred = clf.predict(X_test)
        train_times.append((end-start))
        acc_scores.append(accuracy_score(y_test, y_pred))
        bal_acc_scores.append(balanced_accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        auroc_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr', average='weighted'))

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # print the average scores and the overall confusion matrix
    print(f'Train time: {np.mean(train_times)}')
    print(f'Accuracy: {np.mean(acc_scores)}')
    print(f'Balanced Accuracy: {np.mean(bal_acc_scores)}')
    print(f'Precision: {np.mean(prec_scores)}')
    print(f'Recall: {np.mean(recall_scores)}')
    print(f'F1 score: {np.mean(f1_scores)}')
    print(f'AUROC score: {np.mean(auroc_scores)}')

    cm = confusion_matrix(y_true_all, y_pred_all)
    print("Confusion Matrix:\n", cm)

    """ save all the results to a txt file """
    with open(f'./{filepath}.txt', 'w') as f:
        f.write("Classifier Results\n")
        f.write("------------------------------------------------\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Classifier: {str(clf)}\n")
        f.write("------------------------------------------------\n")
        f.write(f"Train time: {np.mean(train_times)}\n")
        f.write(f"Accuracy: {np.mean(acc_scores)}\n")
        f.write(f"Balanced Accuracy: {np.mean(bal_acc_scores)}\n")
        f.write(f"Precision: {np.mean(prec_scores)}\n")
        f.write(f"Recall: {np.mean(recall_scores)}\n")
        f.write(f"F1 Score: {np.mean(f1_scores)}\n")
        f.write(f"AUROC: {np.mean(auroc_scores)}\n")
        f.write(f"Confusion Matrix:\n {cm}\n")
        f.write("------------------------------------------------\n")



def col_ensemble_experiment(X, y, clf_to_use, filepath, dataset_name):

    rows, cols = np.shape(X)

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


    time_series_experiment(X, y, clf, filepath, dataset_name)



if __name__ == "__main__":
    # main()
    test2()