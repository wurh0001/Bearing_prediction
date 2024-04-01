'''
Date: 2024-03-31 20:20:21
LastEditors: wurh2022 z02014268@stu.ahu.edu.cn
LastEditTime: 2024-04-01 13:51:34
FilePath: \Bearing_prediction\predict.py
Description: Do not edit
'''

import csv
import os
import pandas as pd

class Dataset():
    def __init__():
        pass


def get_data(bearing_path):
    data = []
    subfiles = os.listdir(bearing_path)
    for subfile in subfiles:
        subfile_name = os.path.join(bearing_path, subfile)
        if os.path.isfile(subfile_name):
            print(subfile_name)
            with open(subfile_name, 'r') as data_file:
                data_reader = csv.reader(data_file, delimiter=',')
                for row in data_reader:
                    data.append(row)
                bearing_columns = ['hour', 'minute', 'second', 'micro', 'Horizontal_acceleration', 'Vertical_acceleration']
                bearing_data = pd.DataFrame(data, columns=bearing_columns)
