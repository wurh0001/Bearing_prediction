'''
Date: 2024-03-31 20:20:21
LastEditors: wurh2022 z02014268@stu.ahu.edu.cn
LastEditTime: 2024-04-05 00:44:42
FilePath: \Bearing_prediction\predict.py
Description: Do not edit
'''

from audioop import rms
import csv
import os
import numpy as np
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
    
    def calculate_data(self, data):
        # 通过data计算特征
        '''===================时域特征========================='''
        # 峰值
        peak_value = data['Horizontal_acceleration'].max()
        # 均方根值
        rms_value = np.sqrt(np.mean(data['Horizontal_acceleration'] ** 2))
        # 方差
        variance = np.var(data['Horizontal_acceleration'])
        # 整流平均值
        rectified_mean = np.mean(np.abs(data['Horizontal_acceleration']))
        # 峰峰值
        peak_to_peak_value = data['Horizontal_acceleration'].max() - data['Horizontal_acceleration'].min()
        # 方根幅值
        rmsa = np.mean(np.sqrt(np.abs(data['Horizontal_acceleration']))) ** 2
        # 峭度
        kurtosis_value = data['Horizontal_acceleration'].kurt()
        # 偏度
        skewness_value = data['Horizontal_acceleration'].skew()
        # 波形因子
        waveform_factor = rms_value / rectified_mean
        # 峰值因子
        peak_factor = peak_value / rms_value
        # 脉冲因子
        impulse_factor = peak_value / rectified_mean
        # 裕度因子
        margin_factor = peak_value / rmsa
        # 能量
        energy = np.sum(data['Horizontal_acceleration'] ** 2)

        '''===================频域特征========================='''

        '''------------------------------傅里叶变换------------------------------------------'''
        # 计算采样频率
        fs = 1 / (data['time'][1] - data['time'][0]).total_seconds()

        # 计算信号的长度
        n = len(data.loc[:, 'Horizontal_acceleration'])

        # 计算频率
        f = np.linspace(0, fs, n)

        # 计算频谱
        fft = np.fft.fft(data['Horizontal_acceleration'])

        # 计算频谱的幅值
        fft_amp = np.abs(fft)
        '''------------------------------频谱特征----------------------------------------------'''
        # 计算平均频率
        f_mean = np.mean(f[np.where(fft_amp > 0.1 * fft_amp.max())])

        # 计算频谱的主频
        f_max = f[np.argmax(fft_amp)]

        # 计算均方根频率
        f_rms = np.sqrt(np.sum(fft_amp**2) / n)

        # 计算频率标准差
        f_std = np.sqrt(np.sum((f - f_mean)**2 * fft_amp**2) / n)

        # 计算中心频率
        f_median = np.sum(f * fft_amp) / np.sum(fft_amp)
        
        '''===================时频域特征========================='''

        '''-------------------------------离散小波变换---------------------------------------------'''
        

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

