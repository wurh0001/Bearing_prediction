'''
Date: 2024-03-31 20:20:21
LastEditors: wurh2022 z02014268@stu.ahu.edu.cn
LastEditTime: 2024-04-05 18:01:59
FilePath: \Bearing_prediction\predict.py
Description: Do not edit
'''

from cgi import test
import os
import csv
import random

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split



# 轴承数据集类
class Bearing_Dataset(Dataset):
    def __init__(self, bearing_path):
        self.bearing_path = bearing_path
        self.subfiles = os.listdir(bearing_path)
        self.length = len(self.subfiles)
        
    def __len__(self):
        return self.length
    
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
            bearing_data = self.calculate_data(bearing_data)
            rul = idx / self.length * 100
            # bearing_data中增加剩余寿命列
            bearing_data['RUL'] = rul
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

        # 构建时域特征向量
        time_domain_feature = [peak_value, rms_value, variance, rectified_mean, 
                                        peak_to_peak_value, rmsa, kurtosis_value, skewness_value, 
                                        waveform_factor, peak_factor, impulse_factor, margin_factor, energy]

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

        # 构建频域特征向量
        frequency_domain_feature = [f_mean, f_max, f_rms, f_std, f_median]
        
        '''===================时频域特征========================='''

        '''-------------------------------离散小波变换---------------------------------------------'''
        

        return time_domain_feature, frequency_domain_feature


# 数据加载器
def bearing_dataloader(bearing_path, batch_size):
    bearing_dataset = Bearing_Dataset(bearing_path)    
    bearing_dataloader = DataLoader(bearing_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 划分训练集和测试集
    train_size = int(0.8 * len(bearing_dataset))
    test_size = len(bearing_dataset) - train_size
    split_size = [train_size, test_size]
    train_dataset, test_dataset = random_split(bearing_dataset, split_size)

    # 创建训练集加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 创建测试集加载器
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


# 轴承寿命预测模型
class Bearing_Predictor():
    # 初始化函数实现加载数据集和数据加载器
    def __init__(self, bearing_path, batch_size):
        self.bearing_path = bearing_path
        self.batch_size = batch_size
        self.bearing_dataloader = bearing_dataloader(self.bearing_path, self.batch_size)
    
    # 
    def predict(self):
        for data in self.bearing_dataloader:
            time_domain_feature, frequency_domain_feature = self.calculate_data(data)
            # 进行预测
            # ...
            # 返回预测结果
            return prediction


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

