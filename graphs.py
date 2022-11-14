import matplotlib.pyplot as plt
import pandas as pd
import os

CURRENT_PATH = os.getcwd()
print(CURRENT_PATH)

def displayGraphs(table, movement):

    table = pd.read_csv(os.path.join(CURRENT_PATH, "data_instance", table))

    a_x = table['a_x']
    a_y = table['a_y']
    a_z = table['a_z']
    g_x = table['g_x']
    g_y = table['g_y']
    g_z = table['g_z']
    x = table['Timestamps (ms)']

    plt.subplot(6, 1, 1)
    plt.plot(x,a_x)
    plt.title("X_acc")
    plt.subplot(6, 1, 2)
    plt.plot(x,a_y)
    plt.title("Y_acc")
    plt.subplot(6, 1, 3)
    plt.plot(x,a_z)
    plt.title("Z_acc")

    plt.subplot(6, 1, 4)
    plt.plot(x,g_x)
    plt.title("X_gyro")
    plt.subplot(6, 1, 5)
    plt.plot(x,g_y)
    plt.title("Y_gyro")
    plt.subplot(6, 1, 6)
    plt.plot(x,g_z)
    plt.title("Z_gyro")

    plt.suptitle(movement[0])
    # plt.show()

    filename = movement[0] + "_" + movement[2].split(".")[0] + ".png"

    print(filename)

    plt.savefig(os.path.join(CURRENT_PATH, "data_graphs", filename))

    plt.clf()

for subdir, dirs, files in os.walk(os.path.join(CURRENT_PATH, "data_instance")):
        for file in files:
            movement = file.split('_')
            # print(movement)
            displayGraphs(file, movement)



