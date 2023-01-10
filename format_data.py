"""
Author: Harry Redman
Created on: 16/12/2022

Description: takes the raw sensor data csv files and combines both the accelerometer and gyroscope data together into one csv and removes the 
first 5 seconds of data and then take a 10 second sample window of the data instance]

"""
import os
import pandas as pd
import numpy as np

def trim_dataset(data, movement):

    if movement in ['benchpress', 'squat', 'deadlift', 'militarypress']:
        """ remove the first 5 seconds of the data recording to reduce noised caused by setting up the exercise"""
        data = data.drop(data.index[range(50)]).reset_index(drop=True)
        print(movement, "5")
    else:
        """ remove the first 3 seconds of the data recording to reduce noised caused by setting up the exercise"""
        data = data.drop(data.index[range(30)]).reset_index(drop=True)
        print(movement, "3")



    """extract 10 seconds of the data and then return this as a data instance"""
    data = data.iloc[0:100]

    data = data[['X','Y','Z']]

    return data




def combine_data(accel_data, gyro_data, movement, instanceNumber, folderPath):

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

    """ normalises data into the range 0 and 1 if option if selected """
    data_instance = (data_instance-data_instance.min())/(data_instance.max()-data_instance.min())

    data_instance.insert(loc=0, column='Timestamps (ms)', value=timestamps)


    filename = f'{folderPath}/{movement}_instance_{instanceNumber}'

    data_instance.to_csv(f'{filename}.csv', index=False)



def main():
    CURRENT_PATH = os.getcwd()

    rootdir = os.path.join(CURRENT_PATH, "Data", "rawData")
    formatted_dir = os.path.join(CURRENT_PATH, "Data", "formatted_data")

    for participant in os.listdir(rootdir):

        """ create participant folder in formatted data instances if it doesn't exist yet """
        if not os.path.exists(os.path.join(formatted_dir, participant)):
            os.makedirs(os.path.join(formatted_dir, participant))
        
        participant_dir = os.path.join(formatted_dir, participant)

        current_participant_dir = os.path.join(rootdir, participant)

        for movement in os.listdir(current_participant_dir):
            # print(movement)

            """ create movement folder in formatted data instances if it doesn't exist yet """
            if not os.path.exists(os.path.join(participant_dir, movement)):
                os.makedirs(os.path.join(participant_dir, movement))
        
            movement_dir = os.path.join(participant_dir, movement)

            current_movement_dir = os.path.join(current_participant_dir, movement)
            instance_number = 1

            for recording in os.listdir(current_movement_dir):
                
                exercise_recordings = os.path.join(current_movement_dir, recording)                
                accel_data = pd.read_csv(os.path.join(exercise_recordings, 'Accelerometer.csv'))
                gyro_data = pd.read_csv(os.path.join(exercise_recordings, 'Gyroscope.csv'))


                """ check recordings have same amount of data points, since one of the recording typically has one or two extra timestamps"""

                if (len(gyro_data) != len(accel_data)):
                    num_recordings = min(len(gyro_data), len(accel_data))
                    accel_data = accel_data.iloc[:num_recordings:]
                    gyro_data = gyro_data.iloc[:num_recordings:]

                accel_data = trim_dataset(accel_data, movement)
                gyro_data = trim_dataset(gyro_data, movement)

            
                combine_data(accel_data, gyro_data, movement, instance_number, movement_dir)

                instance_number+= 1

if __name__ == "__main__":
    main()

