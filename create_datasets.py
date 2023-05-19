"""
Author: Harry Redman
Created on: 11/01/23
Description: takes csv files of each data instance and creates full datasets files
with no train test split, in a .ts file format so that it can be used with classifiers

"""
import pandas as pd
import numpy as np
import os
from sktime.datasets._data_io import write_ndarray_to_tsfile


def create_ts_file(path_to_save, univariate_data_set=False, axis=None, multivariate_data='all'):   
    CURRENT_PATH = os.getcwd()

    rootdir = os.path.join(CURRENT_PATH, "Data", "formatted_data")

    for participant in os.listdir(rootdir):


        if participant != "Harry":
            continue

        """ getting different datasets"""

        X_gym = []
        y_gym = []        


        X_weighted = []
        y_weighted = []        

        X_body_weight = []
        y_body_weight = []                    

        current_participant_dir = os.path.join(rootdir, participant)

        for movement in os.listdir(current_participant_dir):


            current_movement_dir = os.path.join(current_participant_dir, movement)

            for data_instance in os.listdir(current_movement_dir):

                class_label = data_instance.split('_')
                data_instance = pd.read_csv(os.path.join(current_movement_dir, data_instance))
                data_instance = data_instance.drop(columns=['Timestamps (ms)'])
                data_instance = np.transpose(data_instance)


                """ only create univariate dataset that includes all exercises"""
                if univariate_data_set:
                    data_instance = data_instance.loc[axis]

                    gym_filename = f"{participant}_gym_movements_{axis[0].replace('_', '')}"

                else:

                    """ Selects axis table depending on what multivariate dataset needs to be created """

                    if multivariate_data == 'accel':
                        data_instance = data_instance.loc[['a_x', 'a_y', 'a_z']]
                        gym_filename = f"{participant}_gym_movements_accel"
                        weighted_filename = f"{participant}_weighted_movements_accel"
                        bodyweight_filename = f"{participant}_bodyweight_movements_accel"
                    elif multivariate_data == 'gyro':
                        data_instance = data_instance.loc[['g_x', 'g_y', 'g_z']]
                        gym_filename = f"{participant}_gym_movements_gyro"
                        weighted_filename = f"{participant}_weighted_movements_gyro"
                        bodyweight_filename = f"{participant}_bodyweight_movements_gyro"
                    else:
                        gym_filename = f"{participant}_gym_movements"
                        weighted_filename = f"{participant}_weighted_movements"
                        bodyweight_filename = f"{participant}_bodyweight_movements"



                X_gym.append(data_instance)
                y_gym.append(class_label[0])

                if movement in ['benchpress', 'squat', 'deadlift', 'militarypress']:
                    X_weighted.append(data_instance)
                    y_weighted.append(class_label[0])

                elif movement in ['pullup', 'pressup', 'situp']:
                    X_body_weight.append(data_instance)
                    y_body_weight.append(class_label[0])


        X_gym = np.asarray(X_gym)
        y_gym = np.asarray(y_gym)

        """ get the unique class labels """
        class_labels = set(y_gym)

        """ create ts file for a 50:50 train and test split"""
        write_ndarray_to_tsfile(data=X_gym, path=f'{path_to_save}/gym', problem_name=f"{gym_filename}", class_label=class_labels,
        class_value_list=y_gym, equal_length=True, series_length=100)

        "only create multivariate datasets for weighted and body weight exercises"
        if not univariate_data_set:


            """ create datasets for weighted exercises """

            X_weighted = np.asarray(X_weighted)
            y_weighted = np.asarray(y_weighted)

            class_labels = set(y_weighted)

            """ create ts file for a 50:50 train and test split"""
            write_ndarray_to_tsfile(data=X_weighted, path=f'{path_to_save}/weighted', problem_name=f"{weighted_filename}", class_label=class_labels,
            class_value_list=y_weighted, equal_length=True, series_length=100)


            """ create datasets for body weight exercises """

            X_body_weight = np.asarray(X_body_weight)
            y_body_weight = np.asarray(y_body_weight)

            class_labels = set(y_body_weight)

            """ create ts file for a 50:50 train and test split"""
            write_ndarray_to_tsfile(data=X_body_weight, path=f'{path_to_save}/bodyweight', problem_name=f"{bodyweight_filename}", class_label=class_labels,
            class_value_list=y_body_weight, equal_length=True, series_length=100)

def main():
    CURRENT_PATH = os.getcwd()
    path_to_save = os.path.join(CURRENT_PATH, "Data", "datasets")

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)



    create_ts_file(path_to_save=path_to_save)

    create_ts_file(path_to_save=path_to_save, multivariate_data='accel')
    create_ts_file(path_to_save=path_to_save, multivariate_data='gyro')

    create_ts_file(path_to_save=path_to_save, univariate_data_set=True, axis=['a_x'])
    create_ts_file(path_to_save=path_to_save, univariate_data_set=True, axis=['a_y'])
    create_ts_file(path_to_save=path_to_save, univariate_data_set=True, axis=['a_z'])

    create_ts_file(path_to_save=path_to_save, univariate_data_set=True, axis=['g_x'])
    create_ts_file(path_to_save=path_to_save, univariate_data_set=True, axis=['g_y'])
    create_ts_file(path_to_save=path_to_save, univariate_data_set=True, axis=['g_z'])
    
    



if __name__ == "__main__":
    main()

