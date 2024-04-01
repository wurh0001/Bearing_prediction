'''
Date: 2024-03-31 20:20:21
LastEditors: wurh2022 z02014268@stu.ahu.edu.cn
LastEditTime: 2024-04-01 23:33:16
FilePath: \Bearing_prediction\predict.py
Description: Do not edit
'''

import csv
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Bearing_Dataset(Dataset):
    def __init__(self, bearing_path):
        self.bearing_path = bearing_path
        self.subfiles = os.listdir(bearing_path)
        # pass
    def __len__(self):
        return len(self.subfiles)
    def __getitem__(self, idx):
        subfile = self.subfiles[idx]
        subfile_name = os.path.join(self.bearing_path, subfile)
        data = []
        with open(subfile_name, 'r') as data_file:
            data_reader = csv.reader(data_file, delimiter=',')
            for row in data_reader:
                data.append(row)
            bearing_columns = ['hour', 'minute', 'second', 'micro', 'Horizontal_acceleration', 'Vertical_acceleration']
            bearing_data = pd.DataFrame(data, columns=bearing_columns)
            bearing_data['hour'] = bearing_data['hour'].astype('int16')
            bearing_data['minute'] = bearing_data['minute'].astype('int16')
            bearing_data['second'] = bearing_data['second'].astype('int16')
            bearing_data['micro'] = bearing_data['micro'].astype('float32')
            bearing_data['Horizontal_acceleration'] = bearing_data['Horizontal_acceleration'].astype('float32')
            bearing_data['Vertical_acceleration'] = bearing_data['Vertical_acceleration'].astype('float32')
            time = pd.to_timedelta(data['hour'], unit='h') + pd.to_timedelta(data['minute'], unit='m') + pd.to_timedelta(data['second'], unit='s') + pd.to_timedelta(data['micro'], unit='us')
            data['time'] = time
            # 删去原有的时间数据，只保留时间戳
            bearing_data = bearing_data.drop(['hour', 'minute', 'second', 'micro'], axis=1)
        return bearing_data

def get_data(bearing_path):
    total_data = []
    subfiles = os.listdir(bearing_path)
    for subfile in subfiles:
        subfile_name = os.path.join(bearing_path, subfile)
        if os.path.isfile(subfile_name):
            print(subfile_name)
            data = []
            with open(subfile_name, 'r') as data_file:
                data_reader = csv.reader(data_file, delimiter=',')
                for row in data_reader:
                    data.append(row)
                bearing_columns = ['hour', 'minute', 'second', 'micro', 'Horizontal_acceleration', 'Vertical_acceleration']
                bearing_data = pd.DataFrame(data, columns=bearing_columns)
                bearing_data['hour'] = bearing_data['hour'].astype('int16')
                bearing_data['minute'] = bearing_data['minute'].astype('int16')
                bearing_data['second'] = bearing_data['second'].astype('int16')
                bearing_data['micro'] = bearing_data['micro'].astype('float32')
                bearing_data['Horizontal_acceleration'] = bearing_data['Horizontal_acceleration'].astype('float32')
                bearing_data['Vertical_acceleration'] = bearing_data['Vertical_acceleration'].astype('float32')
                time = pd.to_timedelta(data['hour'], unit='h') + pd.to_timedelta(data['minute'], unit='m') + pd.to_timedelta(data['second'], unit='s') + pd.to_timedelta(data['micro'], unit='us')
                data['time'] = time
                # 删去原有的时间数据，只保留时间戳
                bearing_data = bearing_data.drop(['hour', 'minute', 'second', 'micro'], axis=1)
            total_data.append(bearing_data)
    return total_data

