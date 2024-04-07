'''
Date: 2024-03-31 20:20:21
LastEditors: wurh2022 z02014268@stu.ahu.edu.cn
LastEditTime: 2024-04-07 23:58:45
FilePath: \Bearing_prediction\predict.py
Description: Do not edit
'''

import os
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
# from torch.nn import Transformer


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
            rul = torch.tensor(rul)
            # bearing_data = torch.cat((bearing_data, rul), 0)
        return bearing_data, rul
    
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
        
        # 将数据类型转换为张量
        time_domain_feature = torch.tensor(time_domain_feature)
        frequency_domain_feature = torch.tensor(frequency_domain_feature)
        # 将时域特征和频域特征拼接成一个特征向量
        features_vetor = torch.cat((time_domain_feature, frequency_domain_feature), 0)
        return features_vetor


# 数据加载器
def bearing_dataloader(bearing_path, batch_size):
    bearing_dataset = Bearing_Dataset(bearing_path)    
    bearing_dataloader = DataLoader(bearing_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 划分训练集和测试集
    # train_size = int(0.8 * len(bearing_dataset))
    # test_size = len(bearing_dataset) - train_size
    # split_size = [train_size, test_size]
    # train_dataset, test_dataset = random_split(bearing_dataset, split_size)

    # 创建训练集加载器
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 创建测试集加载器
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return bearing_dataloader


# 轴承寿命预测模型
class Bearing_Predictor(nn.model, nn.Transformer):
    # 初始化函数搭建transformer网络
    def __init__(self, feature_dim, embedding_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation):
        # super(Bearing_Predictor, self).__init__()
        # 调用父类的初始化函数
        nn.model.__init__(self)
        nn.Transformer.__init__(self, d_model=embedding_dim, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = None
        # embedding层用于将输入的特征进行更深层次的抽象
        self.input_embedding = nn.Embedding(feature_dim, embedding_dim=embedding_dim)
        # 编码层使用transformerencoder，解码层使用简单的全连接层
        # TODO: 使用transformer解码层
        self.decoder = nn.Squential(nn.Linear(feature_dim, feature_dim),
                                    # nn.ReLU(),
                                    nn.Linear(feature_dim, 1),
                                    nn.Sigmoid())
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder[0].bias)
        nn.init.zeros_(self.decoder[2].bias)
        nn.init.uniform_(self.decoder[0].weight, -initrange, initrange)
        nn.init.uniform_(self.decoder[2].weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                device = src.device
                mask = self.generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        src = self.input_embedding(src)
        output = self.encoder(src, self.src_mask)
        output = self.decoder(output)
        # return F.log_softmax(output, dim=-1)
        return output


# 模型前向传播
def model_forward(batch, model, criterion, device):
    features, rul = batch
    features = features.to(device)
    rul = rul.to(device)
    output = model(features)
    loss = criterion(output, rul)
    accuracy = output - rul

    return loss, accuracy


# 训练模型
def train_model(model, train_dataloader, criterion, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss, accuracy = model_forward(batch, model, criterion, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
        print('Epoch: %d, Loss: %.4f, Accuracy: %.4f' % (epoch, total_loss, total_accuracy))
        test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)
        print('Test Loss: %.4f, Test Accuracy: %.4f' % (test_loss, test_accuracy))



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

