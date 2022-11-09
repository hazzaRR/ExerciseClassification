import os

CURRENT_PATH = os.getcwd()
os.path.join(CURRENT_PATH, "rawData")
rootdir = os.path.join(CURRENT_PATH, "rawData")

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(dirs)
        # print(os.path.join(subdir, file))