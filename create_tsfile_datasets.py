"""
Author: Harry Redman
Created on: 11/01/23
Description: [takes csv files of each data instance and creates datasets in a .ts file format so that it can be used with classifiers]
"""
import pandas as pd
import numpy as np
import os
from sktime.datasets._data_io import write_ndarray_to_tsfile
import math


def create_ts_file(path_to_save):   
    CURRENT_PATH = os.getcwd()

    rootdir = os.path.join(CURRENT_PATH, "Data", "formatted_data")

    for participant in os.listdir(rootdir):

        X_train_gym = []
        y_train_gym = []        
        X_test_gym = []
        y_test_gym = []


        X_train_weighted = []
        y_train_weighted = []        
        X_test_weighted = []
        y_test_weighted = []

        X_train_body_weight = []
        y_train_body_weight = []        
        X_test_body_weight = []
        y_test_body_weight = []              

        current_participant_dir = os.path.join(rootdir, participant)

        for movement in os.listdir(current_participant_dir):


            current_movement_dir = os.path.join(current_participant_dir, movement)

            movement_count = len(os.listdir(current_movement_dir))
            current_count = 0

            for data_instance in os.listdir(current_movement_dir):

                class_label = data_instance.split('_')
                data_instance = pd.read_csv(os.path.join(current_movement_dir, data_instance))
                data_instance = data_instance.drop(columns=['Timestamps (ms)'])
                data_instance = np.transpose(data_instance)

                if current_count < math.floor(movement_count/2):
                    X_train_gym.append(data_instance)
                    y_train_gym.append(class_label[0])

                    if movement in ['benchpress', 'squat', 'deadlift', 'militarypress']:
                        X_train_weighted.append(data_instance)
                        y_train_weighted.append(class_label[0])
                    else:
                        X_train_body_weight.append(data_instance)
                        y_train_body_weight.append(class_label[0])

                else:
                    X_test_gym.append(data_instance)
                    y_test_gym.append(class_label[0])

                    if movement in ['benchpress', 'squat', 'deadlift', 'militarypress']:
                        X_test_weighted.append(data_instance)
                        y_test_weighted.append(class_label[0])
                    else:
                        X_test_body_weight.append(data_instance)
                        y_test_body_weight.append(class_label[0])

                current_count+= 1




        X_train_gym = np.asarray(X_train_gym)
        X_test_gym = np.asarray(X_test_gym)
        y_train_gym = np.asarray(y_train_gym)
        y_test_gym = np.asarray(y_test_gym)

        """ get the unique class labels """
        class_labels = set(np.concatenate((y_train_gym, y_test_gym), axis=None))

        """ create ts file for a 50:50 train and test split"""
        write_ndarray_to_tsfile(data=X_train_gym, path=path_to_save, problem_name=f"{participant}_gym_movements", class_label=class_labels,
        class_value_list=y_train_gym, equal_length=True, series_length=100, fold="_TRAIN")

        write_ndarray_to_tsfile(data=X_test_gym, path=path_to_save, problem_name=f"{participant}_gym_movements", class_label=class_labels,
        class_value_list=y_test_gym, equal_length=True, series_length=100, fold="_TEST")


        """ create datasets for weighted exercises """

        X_train_weighted = np.asarray(X_train_weighted)
        X_test_weighted = np.asarray(X_test_weighted)
        y_train_weighted = np.asarray(y_train_weighted)
        y_test_weighted = np.asarray(y_test_weighted)

        class_labels = set(np.concatenate((y_train_weighted, y_test_weighted), axis=None))

        """ create ts file for a 50:50 train and test split"""
        write_ndarray_to_tsfile(data=X_train_weighted, path=path_to_save, problem_name=f"{participant}_weighted_movements", class_label=class_labels,
        class_value_list=y_train_weighted, equal_length=True, series_length=100, fold="_TRAIN")

        write_ndarray_to_tsfile(data=X_test_weighted, path=path_to_save, problem_name=f"{participant}_weighted_movements", class_label=class_labels,
        class_value_list=y_test_weighted, equal_length=True, series_length=100, fold="_TEST")


        """ create datasets for body weight exercises """

        X_train_body_weight = np.asarray(X_train_body_weight)
        X_test_body_weight = np.asarray(X_test_body_weight)
        y_train_body_weight = np.asarray(y_train_body_weight)
        y_test_body_weight = np.asarray(y_test_body_weight)

        class_labels = set(np.concatenate((y_train_body_weight, y_test_body_weight), axis=None))

        """ create ts file for a 50:50 train and test split"""
        write_ndarray_to_tsfile(data=X_train_body_weight, path=path_to_save, problem_name=f"{participant}_body_weight_movements", class_label=class_labels,
        class_value_list=y_train_body_weight, equal_length=True, series_length=100, fold="_TRAIN")

        write_ndarray_to_tsfile(data=X_test_body_weight, path=path_to_save, problem_name=f"{participant}_body_weight_movements", class_label=class_labels,
        class_value_list=y_test_body_weight, equal_length=True, series_length=100, fold="_TEST")

def main():
    CURRENT_PATH = os.getcwd()
    path_to_save = os.path.join(CURRENT_PATH, "Data", "datasets")

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)





    create_ts_file(path_to_save)


if __name__ == "__main__":
    main()

