import pandas as pd
import numpy as np
import os
from sktime.datasets._data_io import write_ndarray_to_tsfile

CURRENT_PATH = os.getcwd()


data_array = []

class_label_values = []

rootdir = os.path.join(CURRENT_PATH, "data_instance")


for subdir, dirs, files in os.walk(rootdir):
    for file in files:

        class_label = file.split('_')
        class_label_values.append(class_label[0])

        data_instance = pd.read_csv(os.path.join(CURRENT_PATH, "data_instance", file))

        data_instance = data_instance.drop(columns=['Timestamps (ms)'])

        data_instance = np.transpose(data_instance)

        # print(data_instance)

        data_array.append(data_instance)


data_array = np.asarray(data_array)
class_label_values = np.asarray(class_label_values)

print(np.shape(data_array))

class_labels = set(class_label_values)


write_ndarray_to_tsfile(data=data_array, path=os.path.join(CURRENT_PATH, "data"), problem_name="Powerlift_movements", class_label=class_labels,
class_value_list=class_label_values, equal_length=True, series_length=100)