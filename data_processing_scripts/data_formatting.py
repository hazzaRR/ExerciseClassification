# Author: Harry Redman
# Created on: 10/11/2022
# Last update: [10/11/2022], [created information gain function and skeleton functions for gini and chi squared]
# Description: [take raw sensor data and combine both the accelerometer and gyroscope data together into one csv and remove the 
# first 5 seconds of data and then take a 10 second sample of the data instance]
import os
import pandas as pd
import numpy as np

CURRENT_PATH = os.getcwd()
print(CURRENT_PATH)

movements = ["benchpress", "deadlift", "squat"]
rootdir = os.path.join(CURRENT_PATH, "rawData")


def trim_dataset(data):
    """ remove the first 5 seconds of the data recording to reduce noised caused by setting up the exercise"""
    data = data.drop(data.index[range(50)]).reset_index(drop=True)


    """extract 10 seconds of the data and then return this as a data instance"""
    data = data.iloc[0:100]

    data = data[['X','Y','Z']]

    return data



def combine_data(accel_data, gyro_data, movement, instanceNumber):

    """ create timestamps for data for a 10 second period with intervals of 100ms"""
    timestamps = np.arange(100, 10100, 100)


    data = {

        'Timestamps (ms)': timestamps,
        'a_x': accel_data['X'],
        'a_y': accel_data['Y'],
        'a_z': accel_data['Z'],
        'g_x': gyro_data['X'],
        'g_y': gyro_data['Y'],
        'g_z': gyro_data['Z'],
        
    }

    data_instance = pd.DataFrame(data)

    filename = f'{movement}_instance_{instanceNumber}'
    print(filename)

    print(instanceNumber)

    data_instance.to_csv(f'./data_instance/{filename}.csv', index=False)


for movement in movements:
    print(movement)

    folder_to_check = os.path.join(rootdir, movement)

    for subdir, dirs, files in os.walk(folder_to_check):

        instance_number = 1

        for name in dirs:
            
            currentfolder = os.path.join(folder_to_check, name)

            accel_data = pd.read_csv(os.path.join(currentfolder, 'Accelerometer.csv'))
            gyro_data = pd.read_csv(os.path.join(currentfolder, 'Gyroscope.csv'))

            accel_data = trim_dataset(accel_data)
            gyro_data = trim_dataset(gyro_data)
        

            combine_data(accel_data, gyro_data, movement, instance_number)

            instance_number+= 1
