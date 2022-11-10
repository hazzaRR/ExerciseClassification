import matplotlib.pyplot as plt
import pandas as pd

squat_acc = pd.read_csv('./rawData/squat/squat_set1_2022-11-04/Accelerometer.csv')
squat_gyro = pd.read_csv('./rawData/squat/squat_set1_2022-11-04/Gyroscope.csv')
squat_acc2 = pd.read_csv('./rawData/squat/squat_set2_2022-11-04/Accelerometer.csv')
squat_gyro2 = pd.read_csv('./rawData/squat/squat_set2_2022-11-04/Gyroscope.csv')
bench_press_acc = pd.read_csv('./rawData/bench_press/bench_press_set1_2022-11-04/Accelerometer.csv')
bench_press_gyro = pd.read_csv('./rawData/bench_press/bench_press_set1_2022-11-04/Gyroscope.csv')
deadlift_acc = pd.read_csv('./rawData/deadlift/deadlift_set1_2022-11-04/Accelerometer.csv')
deadlift_gyro = pd.read_csv('./rawData/deadlift/deadlift_set1_2022-11-04/Gyroscope.csv')


def displayGraphs(table):
    y = table['X']
    y1 = table['Y']
    y2 = table['Z']
    x = table['Milliseconds']

    plt.subplot(3, 1, 1)
    plt.plot(x,y)
    plt.title("X Axis")
    plt.subplot(3, 1, 2)
    plt.plot(x,y1)
    plt.title("Y Axis")
    plt.subplot(3, 1, 3)
    plt.plot(x,y2)
    plt.title("Z Axis")

    plt.suptitle("MY SHOP")
    plt.show()


displayGraphs(squat_acc)
displayGraphs(squat_acc2)
displayGraphs(squat_gyro)
displayGraphs(squat_gyro2)
displayGraphs(bench_press_acc)
displayGraphs(bench_press_gyro)
displayGraphs(deadlift_acc)
displayGraphs(deadlift_gyro)