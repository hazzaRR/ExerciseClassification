import matplotlib.pyplot as plt
import pandas as pd
import os

CURRENT_PATH = os.getcwd()
print(CURRENT_PATH)

def displayGraphs(table, movement):

    table = pd.read_csv(os.path.join(CURRENT_PATH, "Prototype", "Prototype_1", "data", "data_instance", table))

    a_x = table['a_x']
    a_y = table['a_y']
    a_z = table['a_z']
    g_x = table['g_x']
    g_y = table['g_y']
    g_z = table['g_z']
    x = table['Timestamps (ms)']

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(x,a_x)
    axs[0, 0].set_title('Accel X Axis')
    axs[1, 0].plot(x,a_y, 'tab:orange')
    axs[1, 0].set_title('Accel Y Axis')
    axs[2, 0].plot(x,a_z, 'tab:red')
    axs[2, 0].set_title('Accel Z Axis')

    axs[0, 1].plot(x,g_x)
    axs[0, 1].set_title('Gyro X Axis')
    axs[1, 1].plot(x,g_y, 'tab:orange')
    axs[1, 1].set_title('Gyro Y Axis')
    axs[2, 1].plot(x,g_z, 'tab:red')
    axs[2, 1].set_title('Gyro Z Axis')

    plt.tight_layout()

    filename = movement[0] + "_" + movement[2].split(".")[0] + ".png"

    print(filename)

    plt.savefig(os.path.join(CURRENT_PATH, "data_graphs", filename))

    plt.clf()


def main():


    for subdir, dirs, files in os.walk(os.path.join(CURRENT_PATH, "Prototype", "Prototype_1", "data", "data_instance")):
            for file in files:
                movement = file.split('_')
                print(movement)
                displayGraphs(file, movement)



if __name__ == "__main__":
    main()