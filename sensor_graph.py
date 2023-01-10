import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def produceGraph(filepath):

    accel_data = pd.read_csv(os.path.join(filepath, 'Accelerometer.csv'))
    gyro_data = pd.read_csv(os.path.join(filepath, 'Gyroscope.csv'))

    if (len(gyro_data) != len(accel_data)):
        num_recordings = min(len(gyro_data), len(accel_data))
        accel_data = accel_data.iloc[:num_recordings:]
        gyro_data = gyro_data.iloc[:num_recordings:]

    print(accel_data.columns)

    timestamps = accel_data['Milliseconds']

    print(timestamps)

    fig, ax = plt.subplots()

    ax.plot(timestamps, accel_data['X'], '-', label="accel_X")
    ax.plot(timestamps, accel_data['Y'], '-', label="accel_Y")
    # ax.plot(timestamps, accel_data['Z'], '-', label="accel_Z")
    # ax.plot(timestamps, gyro_data['X'], '-', label="gyro_X")
    # ax.plot(timestamps, gyro_data['Y'], '-', label="gyro_Y")
    # ax.plot(timestamps, gyro_data['Z'], '-', label="gyro_Z")

    ax.legend()
    plt.show()

def main():
    CURRENT_PATH = os.getcwd()
    filepath = os.path.join(CURRENT_PATH, 'Data', 'rawData', 'Harry', 'pullup', 'pullup_set11_2023-01-05')
    # filepath = os.path.join(CURRENT_PATH, 'Data', 'rawData', 'Harry', 'pressup', 'pressup_set2_2022-12-28')
    # filepath = os.path.join(CURRENT_PATH, 'Data', 'rawData', 'Harry', 'situp', 'situp_set2_2022-12-28')
    # filepath = os.path.join(CURRENT_PATH, 'Data', 'rawData', 'Harry', 'benchpress', 'benchpress_set4_2022-11-04')
    produceGraph(filepath)

if __name__ == "__main__":
    main()