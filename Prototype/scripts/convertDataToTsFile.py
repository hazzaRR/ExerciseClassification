# Author: Harry Redman
# Created on: 10/11/2022
# Last update: [10/11/2022], [transforms data from csv file to ts file for powerlift movements]
# Description: [takes csv files of each data instance and converts all the data into a .ts file so that it can be used with classifiers]

import pandas as pd
import numpy as np
import os
from sktime.datasets._data_io import write_ndarray_to_tsfile


def create_ts_file(data_path, path_to_save, test_split=False):   
    data_array = []
    class_label_values = []


    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            
            class_label = file.split('_')
            class_label_values.append(class_label[0])

            data_instance = pd.read_csv(os.path.join(data_path, file))

            data_instance = data_instance.drop(columns=['Timestamps (ms)'])

            data_instance = np.transpose(data_instance)

            data_array.append(data_instance)


    data_array = np.asarray(data_array)
    class_label_values = np.asarray(class_label_values)

    class_labels = set(class_label_values)

    if test_split == True:
        bp_train = np.arange(0,12,1)
        deadlift_train = np.arange(17,29,1)
        squat_train = np.arange(33,45,1)

        bp_test = np.arange(12,17,1)
        deadlift_test = np.arange(29,33,1)
        squat_test = np.arange(45,52,1)

        train_value_indexes = np.concatenate((bp_train, deadlift_train, squat_train), axis=None)
        test_value_indexes = np.concatenate((bp_test, deadlift_test, squat_test), axis=None)


        X_train = data_array[train_value_indexes]
        y_train = class_label_values[train_value_indexes]

        X_test = data_array[test_value_indexes]
        y_test = class_label_values[test_value_indexes]

        write_ndarray_to_tsfile(data=X_train, path=path_to_save, problem_name="Powerlift_movements", class_label=class_labels,
        class_value_list=y_train, equal_length=True, series_length=100, fold="_TRAIN")

        write_ndarray_to_tsfile(data=X_test, path=path_to_save, problem_name="Powerlift_movements", class_label=class_labels,
        class_value_list=y_test, equal_length=True, series_length=100, fold="_TEST")
    else:
        write_ndarray_to_tsfile(data=data_array, path=path_to_save, problem_name="Powerlift_movements", class_label=class_labels,
        class_value_list=class_label_values, equal_length=True, series_length=100)



def main():
    CURRENT_PATH = os.getcwd()
    # rootdir = os.path.join(CURRENT_PATH, "Prototype", "Prototype_1", "data", "data_instance")
    # path_to_save = os.path.join(CURRENT_PATH, "Prototype", "Prototype_1", "data")

    rootdir = os.path.join(CURRENT_PATH, "Prototype", "Prototype_2", "data", "data_instance_normalised")
    path_to_save = os.path.join(CURRENT_PATH, "Prototype", "Prototype_2", "data")

    create_ts_file(data_path=rootdir, path_to_save=path_to_save,  test_split=True)


if __name__ == "__main__":
    main()

