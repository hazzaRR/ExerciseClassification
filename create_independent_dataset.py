"""
Author: Harry Redman


Description: accepts csv files from each data instance and generates a train-test dataset in.ts file format,
 consisting of Harry's data in the training file and Barnaby's data in the test file, to be used with classifiers in person-independent experiments.

"""
import pandas as pd
import numpy as np
import os
from sktime.datasets._data_io import write_ndarray_to_tsfile


def create_ts_file(path_to_save):   
    CURRENT_PATH = os.getcwd()

    rootdir = os.path.join(CURRENT_PATH, "Data", "formatted_data")

    
    X_train = []
    y_train = []        
    X_test = []
    y_test = []         


    for participant in os.listdir(rootdir):
    
        current_participant_dir = os.path.join(rootdir, participant)

        for movement in os.listdir(current_participant_dir):

            current_movement_dir = os.path.join(current_participant_dir, movement)


            for data_instance in os.listdir(current_movement_dir):
                class_label = data_instance.split('_')
                data_instance = pd.read_csv(os.path.join(current_movement_dir, data_instance))
                data_instance = data_instance.drop(columns=['Timestamps (ms)'])
                data_instance = np.transpose(data_instance)

                if participant == "Harry":
                    X_train.append(data_instance)
                    y_train.append(class_label[0])
                else:
                    X_test.append(data_instance)
                    y_test.append(class_label[0])



    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # """ get the unique class labels """
    class_labels = set(np.concatenate((y_train, y_test), axis=None))

    print(class_labels)

    """ create ts file for a train and test split """
    write_ndarray_to_tsfile(data=X_train, path=f'{path_to_save}', problem_name="person_independent", class_label=class_labels,
    class_value_list=y_train, equal_length=True, series_length=100, fold="_TRAIN")

    write_ndarray_to_tsfile(data=X_test, path=f'{path_to_save}', problem_name="person_independent", class_label=class_labels,
    class_value_list=y_test, equal_length=True, series_length=100, fold="_TEST")


def main():
    CURRENT_PATH = os.getcwd()
    path_to_save = os.path.join(CURRENT_PATH, "Data", "datasets")

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)



    create_ts_file(path_to_save=path_to_save)

    
    



if __name__ == "__main__":
    main()

