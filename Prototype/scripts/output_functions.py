def save_results(filepath, accel_gyro_acc, accel_gyro_time, accel_acc, accel_time, gyro_acc, gyro_time, \
    accel_x_acc, accel_x_time, accel_y_acc, accel_y_time, accel_z_acc, accel_z_time, gyro_x_acc, gyro_x_time,\
        gyro_y_acc, gyro_y_time, gyro_z_acc, gyro_z_time):

    with open(f'./{filepath}.txt', 'w') as f:
        f.write("Multivariate Results:\n")
        f.write(f" - Accelerometer and Gyroscope Accuracy = {accel_gyro_acc} and train time = {accel_gyro_time}\n")
        f.write(f" - Accelerometer Accuracy = {accel_acc} and train time = {accel_time}\n")
        f.write(f" - Gyroscope Accuracy = {gyro_acc} and train time = {gyro_time}\n")
        f.write("Univariate Results:\n")
        f.write(f" - Accelerometer X axis Accuracy = {accel_x_acc} and train time = {accel_x_time}\n")
        f.write(f" - Accelerometer Y axis Accuracy = {accel_y_acc} and train time = {accel_y_time}\n")
        f.write(f" - Accelerometer Z axis Accuracy = {accel_z_acc} and train time = {accel_z_time}\n")
        f.write(f" - Gyroscope X axis Accuracy = {gyro_x_acc} and train time = {gyro_x_time}\n")
        f.write(f" - Gyroscope Y axis Accuracy = {gyro_y_acc} and train time = {gyro_y_time}\n")
        f.write(f" - Gyroscope Z axis Accuracy = {gyro_z_acc} and train time = {gyro_z_time}\n")

def print_results(accel_gyro_acc, accel_gyro_time, accel_acc, accel_time, gyro_acc, gyro_time, \
    accel_x_acc, accel_x_time, accel_y_acc, accel_y_time, accel_z_acc, accel_z_time, gyro_x_acc, gyro_x_time,\
        gyro_y_acc, gyro_y_time, gyro_z_acc, gyro_z_time):

    print("Multivariate Results:")
    print(f" - Accelerometer and Gyroscope Accuracy = {accel_gyro_acc} and train time = {accel_gyro_time}")
    print(f" - Accelerometer Accuracy = {accel_acc} and train time = {accel_time}")
    print(f" - Gyroscope Accuracy = {gyro_acc} and train time = {gyro_time}")
    print("Univariate Results:")
    print(f" - Accelerometer X axis Accuracy = {accel_x_acc} and train time = {accel_x_time}")
    print(f" - Accelerometer Y axis Accuracy = {accel_y_acc} and train time = {accel_y_time}")
    print(f" - Accelerometer Z axis Accuracy = {accel_z_acc} and train time = {accel_z_time}")
    print(f" - Gyroscope X axis Accuracy = {gyro_x_acc} and train time = {gyro_x_time}")
    print(f" - Gyroscope Y axis Accuracy = {gyro_y_acc} and train time = {gyro_y_time}")
    print(f" - Gyroscope Z axis Accuracy = {gyro_z_acc} and train time = {gyro_z_time}")