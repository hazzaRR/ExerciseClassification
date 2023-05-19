"""
Author: Harry Redman


Description: file used to plot graphs of sensor data of specific data instances 

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def produceGraph(filepath):

    data = pd.read_csv(os.path.join(filepath, 'Accelerometer.csv'))
    gyro_data = pd.read_csv(os.path.join(filepath, 'Gyroscope.csv'))

    if (len(gyro_data) != len(data)):
        num_recordings = min(len(gyro_data), len(data))
        data = data.iloc[:num_recordings:]
        gyro_data = gyro_data.iloc[:num_recordings:]

    print(data.columns)

    timestamps = data['Milliseconds']

    print(timestamps)

    fig, ax = plt.subplots()

    ax.plot(timestamps, data['X'], '-', label="accel_X")
    ax.plot(timestamps, data['Y'], '-', label="accel_Y")
    ax.plot(timestamps, data['Z'], '-', label="accel_Z")
    ax.plot(timestamps, gyro_data['X'], '-', label="gyro_X")
    ax.plot(timestamps, gyro_data['Y'], '-', label="gyro_Y")
    ax.plot(timestamps, gyro_data['Z'], '-', label="gyro_Z")

    ax.legend()
    plt.show()


def produceGraph_instance(filepath):

    data = pd.read_csv(filepath)


    timestamps = data['Timestamps (ms)']

    print(timestamps)

    fig, ax = plt.subplots()

    ax.plot(timestamps, data['a_x'], '-', label="accel_X")
    ax.plot(timestamps, data['a_y'], '-', label="accel_Y")
    ax.plot(timestamps, data['a_z'], '-', label="accel_Z")
    # ax.plot(timestamps, data['g_x'], '-', label="gyro_X")
    # ax.plot(timestamps, data['g_y'], '-', label="gyro_Y")
    # ax.plot(timestamps, data['g_z'], '-', label="gyro_Z")

    ax.legend()
    plt.show()

def main():
    CURRENT_PATH = os.getcwd()
    # filepath = os.path.join(CURRENT_PATH, 'Data', 'rawData', 'Harry', 'pullup', 'pullup_set11_2023-01-05')
    # filepath = os.path.join(CURRENT_PATH, 'Data', 'rawData', 'Harry', 'pressup', 'pressup_set2_2022-12-28')
    # filepath = os.path.join(CURRENT_PATH, 'Data', 'rawData', 'Harry', 'situp', 'situp_set2_2022-12-28')
    # filepath = os.path.join(CURRENT_PATH, 'Data', 'rawData', 'Harry', 'benchpress', 'benchpress_set4_2022-11-04')
    # produceGraph(filepath)

    filepath1 = os.path.join(CURRENT_PATH, 'Data', 'formatted_data', 'Harry', 'deadlift', 'deadlift_instance_10.csv')
    filepath2 = os.path.join(CURRENT_PATH, 'Data', 'formatted_data', 'Barns', 'deadlift', 'deadlift_instance_10.csv')
    produceGraph_instance(filepath1)
    produceGraph_instance(filepath2)

if __name__ == "__main__":
    main()