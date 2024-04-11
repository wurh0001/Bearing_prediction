'''
Date: 2024-03-31 20:20:21
LastEditors: wurh2022 z02014268@stu.ahu.edu.cn
LastEditTime: 2024-04-11 23:40:46
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
from tqdm import tqdm, trange


# 轴承数据集类
class Bearing_Dataset(Dataset):
    def __init__(self, bearing_path):
        self.bearing_path = bearing_path
        self.subfiles = os.listdir(bearing_path)
        self.length = len(self.subfiles)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 数据形状为batch_size x 5 x 18
        # 取五个时间序列数据为一个步长
        subfiles = self.subfiles[idx:idx+10]
        bearing_data_sequence = []
        rul_sequence = []
        i = 0
        for subfile in subfiles:
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
                time = pd.to_timedelta(bearing_data['hour'], unit='h') + pd.to_timedelta(bearing_data['minute'], unit='m') + pd.to_timedelta(bearing_data['second'], unit='s') + pd.to_timedelta(bearing_data['micro'], unit='us')
                bearing_data['time'] = time
                # 删去原有的时间数据，只保留时间戳
                bearing_data = bearing_data.drop(['hour', 'minute', 'second', 'micro'], axis=1)
                bearing_data = self.calculate_data(bearing_data)
                # print(bearing_data)
                # 对特征进行normalization操作       - 选用均值和方差进行归一化
                bearing_data = (bearing_data - bearing_data.mean()) / bearing_data.std()
                rul = (idx + i) / self.length
                i += 1
                # 
                # print(bearing_data.shape, rul.shape)
                # bearing_data = torch.cat((bearing_data, rul), 0)
            bearing_data_sequence.append(bearing_data)
            rul_sequence.append(rul)
        bearing_data_sequence = np.stack(bearing_data_sequence, axis=0)
        bearing_data_sequence = torch.tensor(bearing_data_sequence)
        # 将bearing_data_sequence由float64转换为float32
        bearing_data_sequence = bearing_data_sequence.float()
        rul_sequence = torch.tensor(rul_sequence)
        # rul转换为5x1的张量
        # rul_sequence = rul_sequence.view(rul_sequence.size(0), 1)
        return bearing_data_sequence, rul_sequence
    
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
    bearing_dataloader = DataLoader(bearing_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
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
class Bearing_Predictor(nn.Module):
    # 初始化函数搭建transformer网络
    def __init__(self, feature_dim, embedding_dim, num_heads, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        # 调用父类的初始化函数
        # nn.Model.__init__(self)
        # nn.Transformer.__init__(self, d_model=feature_dim, nhead=num_heads, num_encoder_layers=num_encoder_layers, 
        #                                 dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.model_type = 'Transformer'
        self.src_mask = None
        # self.pos_encoder = None
        self.prenet = nn.Linear(feature_dim, feature_dim)
        # embedding层用于将输入的特征进行更深层次的抽象
        # self.input_embedding = nn.Embedding(feature_dim, embedding_dim=embedding_dim)
        # 编码层使用transformerencoder，解码层使用简单的全连接层
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=num_encoder_layers)
        # TODO: 使用transformer解码层
        self.decoder = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                    nn.ReLU(),
                                    nn.Linear(feature_dim, 10),
                                    # nn.Linear(10, 1),
                                    # nn.Sigmoid()
                                    )
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.input_embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder[0].bias)
        # nn.init.zeros_(self.decoder[1].bias)
        nn.init.uniform_(self.decoder[0].weight, -initrange, initrange)
        # nn.init.uniform_(self.decoder[1].weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                device = src.device
                mask = self.generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        # src = self.input_embedding(src)
        out = self.prenet(src)
        out = out.permute(1, 0, 2)
        output = self.encoder(out, self.src_mask)
        output = output.transpose(0, 1)
        stats = output.mean(dim=1)
        output = self.decoder(stats)
        # return F.log_softmax(output, dim=-1)
        return output


# 模型前向传播
def model_forward(features, rul, model, criterion, device):
    # features, rul = batch
    # features = features.float()  # 将输入数据转换为Float类型
    # 特征为4x18的张量，将其转换为4x1x18的张量
    # features = features.view(features.size(0), 1, features.size(1))
    # print("一个批次中特征的维度：", features.shape)
    # 更改rul的形状为batch_sizex5x1
    rul = rul.view(rul.size(0), 1, rul.size(1))
    # print(rul.shape)
    features = features.to(device)
    rul = rul.to(device)
    output = model(features)
    loss = criterion(output, rul)
    # 计算精度
    accuracy = (output - rul).abs() .float().mean()

    return loss, accuracy


# 训练模型
def train_model(model, train_dataloader, criterion, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):         
        model.train()
        # total_loss = 0
        # total_accuracy = 0
        loop = tqdm((train_dataloader), total=len(train_dataloader))
        for src, rul in loop:
            # 前向传播
            loss, accuracy = model_forward(src, rul, model, criterion, device)
            batch_loss = loss.item()
            batch_accuracy = accuracy.item()
            # 反向传播
            loss.backward()
            # 更新
            optimizer.step()
            optimizer.zero_grad()
            # total_loss += loss.item()
            # total_accuracy += accuracy.mean().item()
            loop.set_description(f'Epoch: 【{epoch}/{epochs}】')
            loop.set_postfix(loss=batch_loss, accuracy=batch_accuracy)

        # print('Epoch: %d, Loss: %.4f, Accuracy: %.4f' % (epoch, total_loss, total_accuracy))
        # test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)
        # print('Test Loss: %.4f, Test Accuracy: %.4f' % (test_loss, test_accuracy))


# 评估模型
def evaluate_model(model, test_dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in test_dataloader:
            loss, accuracy = model_forward(batch, model, criterion, device)
            total_loss += loss.item()
            total_accuracy += accuracy.item()
    return total_loss, total_accuracy


def main():
    # 超参数
    batch_size = 16
    embedding_dim = 256
    num_heads = 2
    num_encoder_layers = 6
    dim_feedforward = 256
    dropout = 0.1
    # activation = 'relu'
    epochs = 10
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集路径
    bearing_path = 'phm-ieee-2012-data-challenge-dataset-master\Learning_set\Bearing1_1'

    # 加载数据集
    v_bearing_dataloader = bearing_dataloader(bearing_path, batch_size)

    # 模型初始化
    model = Bearing_Predictor(feature_dim=18, embedding_dim=embedding_dim, num_heads=num_heads, 
                                        num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, 
                                        dropout=dropout)

    # 损失函数
    criterion = nn.MSELoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    train_model(model, v_bearing_dataloader, criterion, optimizer, device, epochs)    

if __name__ == '__main__':
    main()