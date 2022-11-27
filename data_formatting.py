"""
Author: Harry Redman
Created on: 10/11/2022

Description: takes the raw sensor data csv files and combines both the accelerometer and gyroscope data together into one csv and removes the 
first 5 seconds of data and then take a 10 second sample window of the data instance]

"""
import os
import pandas as pd
import numpy as np

CURRENT_PATH = os.getcwd()

movements = ["benchpress", "deadlift", "squat"]
rootdir = os.path.join(CURRENT_PATH, "rawData")


def trim_dataset(data):
    """ remove the first 5 seconds of the data recording to reduce noised caused by setting up the exercise"""
    data = data.drop(data.index[range(50)]).reset_index(drop=True)


    """extract 10 seconds of the data and then return this as a data instance"""
    data = data.iloc[0:100]

    data = data[['X','Y','Z']]

    return data




def combine_data(accel_data, gyro_data, movement, instanceNumber, folderPath, normalise_data=True):

    """ create timestamps for data for a 10 second period with intervals of 100ms"""
    timestamps = np.arange(100, 10100, 100)

    data = {
        'a_x': accel_data['X'],
        'a_y': accel_data['Y'],
        'a_z': accel_data['Z'],
        'g_x': gyro_data['X'],
        'g_y': gyro_data['Y'],
        'g_z': gyro_data['Z'],
        
    }

    data_instance = pd.DataFrame(data)

    """ normalises data into the range 0 and 1 if option is selected"""
    if normalise_data:
        data_instance = (data_instance-data_instance.min())/(data_instance.max()-data_instance.min())

    data_instance.insert(loc=0, column='Timestamps (ms)', value=timestamps)


    filename = f'{folderPath}/{movement}_instance_{instanceNumber}'

    data_instance.to_csv(f'{filename}.csv', index=False)


for movement in movements:

    folder_to_check = os.path.join(rootdir, movement)

    for subdir, dirs, files in os.walk(folder_to_check):

        instance_number = 1

        for name in dirs:
            
            currentfolder = os.path.join(folder_to_check, name)

            accel_data = pd.read_csv(os.path.join(currentfolder, 'Accelerometer.csv'))
            gyro_data = pd.read_csv(os.path.join(currentfolder, 'Gyroscope.csv'))

            accel_data = trim_dataset(accel_data)
            gyro_data = trim_dataset(gyro_data)
        

            combine_data(accel_data, gyro_data, movement, instance_number, "./Prototype/data/data_instance_normalised")

            instance_number+= 1

