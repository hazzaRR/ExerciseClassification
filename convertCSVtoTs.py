import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

df_gyro = pd.read_csv('./rawData/Gyroscope.csv')
df_acce = pd.read_csv('./rawData/Accelerometer.csv')

for df in [df_gyro, df_acce]:
    df.index = pd.to_datetime(df['time'], unit = 'ns')

df = df_gyro.join(df_acce, lsuffix = '_gyro', rsuffix = '_acce', how = 'outer')#.interpolate()

# df = df.resample('1T').median()

# df = np.interp(df_acce.index, df_gyro.index, df_gyro['x'])


print(df)


# print(df[['seconds_elapsed_acce', 'seconds_elapsed_gyro']])

# df.to_csv('./data/instance1.csv')



# df = pd.read_csv('./data/instance1.csv')[::2]

# df = df.drop(columns=['time_gyro', 'time_acce', 'seconds_elapsed_gyro', 'seconds_elapsed_acce'])

# print(df.reset_index(drop=True))

# print(df)