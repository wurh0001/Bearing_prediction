"""
Date: 2024-03-31 20:20:21
LastEditors: wurh2022 z02014268@stu.ahu.edu.cn
LastEditTime: 2024-04-17 17:19:09
FilePath: \Bearing_prediction\predict.py
Description:  
"""

import os
import csv
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm, trange
from datetime import datetime


# 轴承数据集类
class Bearing_Dataset(Dataset):
    def __init__(self, bearing_path):
        self.bearing_path = bearing_path
        subfiles = os.listdir(bearing_path)
        self.length = len(subfiles) - 10 + 1
        self.data_sequence = []
        for index, subfile in enumerate(subfiles):
            subfile_name = os.path.join(self.bearing_path, subfile)
            data = []
            with open(subfile_name, "r") as data_file:
                data_reader = csv.reader(data_file, delimiter=",")
                for row in data_reader:
                    data.append(row)
                bearing_columns = [
                    "hour",
                    "minute",
                    "second",
                    "micro",
                    "Horizontal_acceleration",
                    "Vertical_acceleration",
                ]
                bearing_data = pd.DataFrame(data, columns=bearing_columns)
                bearing_data["hour"] = bearing_data["hour"].astype("int16")
                bearing_data["minute"] = bearing_data["minute"].astype("int16")
                bearing_data["second"] = bearing_data["second"].astype("int16")
                bearing_data["micro"] = bearing_data["micro"].astype("float32")
                bearing_data["Horizontal_acceleration"] = bearing_data[
                    "Horizontal_acceleration"
                ].astype("float32")
                bearing_data["Vertical_acceleration"] = bearing_data[
                    "Vertical_acceleration"
                ].astype("float32")
                time = (
                    pd.to_timedelta(bearing_data["hour"], unit="h")
                    + pd.to_timedelta(bearing_data["minute"], unit="m")
                    + pd.to_timedelta(bearing_data["second"], unit="s")
                    + pd.to_timedelta(bearing_data["micro"], unit="us")
                )
                bearing_data["time"] = time
                # 删去原有的时间数据，只保留时间戳
                bearing_data = bearing_data.drop(
                    ["hour", "minute", "second", "micro"], axis=1
                )
                bearing_data["rul"] = index / self.length
            self.data_sequence.append(bearing_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 数据形状为batch_size x seq_len x 18
        # 取五个时间序列数据为一个步长
        bearing_data_seq = self.data_sequence[idx : idx + 10]
        bearing_data_sequence = []
        rul_sequence = []
        i = 0
        for sub_bearing_data in bearing_data_seq:
            bearing_data = self.calculate_data(sub_bearing_data)
            # print(bearing_data)
            # 对特征进行normalization操作       - 选用均值和方差进行归一化
            bearing_data = (bearing_data - bearing_data.mean()) / bearing_data.std()
            # bearing_data = torch.tensor(bearing_data)
            rul = sub_bearing_data.loc[1, "rul"]
            # print(bearing_data.shape, rul.shape)
            # bearing_data = torch.cat((bearing_data, rul), 0)
            bearing_data_sequence.append(bearing_data)
            # bearing_data_sequence = torch.tensor(bearing_data_sequence)
            rul_sequence.append(rul)
        # bearing_data_sequence = np.stack(bearing_data_sequence, axis=0)
        bearing_data_sequence = torch.stack(bearing_data_sequence, dim=0)
        # bearing_data_sequence = torch.tensor(bearing_data_sequence)
        # 将bearing_data_sequence由float64转换为float32
        bearing_data_sequence = bearing_data_sequence.float()
        rul_sequence = torch.tensor(rul_sequence)
        rul_sequence = rul_sequence.float()
        # rul转换为sequence_lengthx1的张量
        rul_sequence = rul_sequence.view(rul_sequence.size(0), 1)
        return bearing_data_sequence, rul_sequence

    def calculate_data(self, data):
        # 通过data计算特征
        """===================时域特征========================="""
        # 峰值
        peak_value = data["Horizontal_acceleration"].max()
        # 均方根值
        rms_value = np.sqrt(np.mean(data["Horizontal_acceleration"] ** 2))
        # 方差
        variance = np.var(data["Horizontal_acceleration"])
        # 整流平均值
        rectified_mean = np.mean(np.abs(data["Horizontal_acceleration"]))
        # 峰峰值
        peak_to_peak_value = (
            data["Horizontal_acceleration"].max()
            - data["Horizontal_acceleration"].min()
        )
        # 方根幅值
        rmsa = np.mean(np.sqrt(np.abs(data["Horizontal_acceleration"]))) ** 2
        # 峭度
        kurtosis_value = data["Horizontal_acceleration"].kurt()
        # 偏度
        skewness_value = data["Horizontal_acceleration"].skew()
        # 波形因子
        waveform_factor = rms_value / rectified_mean
        # 峰值因子
        peak_factor = peak_value / rms_value
        # 脉冲因子
        impulse_factor = peak_value / rectified_mean
        # 裕度因子
        margin_factor = peak_value / rmsa
        # 能量
        energy = np.sum(data["Horizontal_acceleration"] ** 2)

        # 构建时域特征向量
        time_domain_feature = [
            peak_value,
            rms_value,
            variance,
            rectified_mean,
            peak_to_peak_value,
            rmsa,
            kurtosis_value,
            skewness_value,
            waveform_factor,
            peak_factor,
            impulse_factor,
            margin_factor,
            energy,
        ]

        """===================频域特征========================="""

        """------------------------------傅里叶变换------------------------------------------"""
        # 计算采样频率
        fs = 1 / (data["time"][1] - data["time"][0]).total_seconds()

        # 计算信号的长度
        n = len(data.loc[:, "Horizontal_acceleration"])

        # 计算频率
        f = np.linspace(0, fs, n)

        # 计算频谱
        fft = np.fft.fft(data["Horizontal_acceleration"])

        # 计算频谱的幅值
        fft_amp = np.abs(fft)
        """------------------------------频谱特征----------------------------------------------"""
        # 计算平均频率
        f_mean = np.mean(f[np.where(fft_amp > 0.1 * fft_amp.max())])

        # 计算频谱的主频
        f_max = f[np.argmax(fft_amp)]

        # 计算均方根频率
        f_rms = np.sqrt(np.sum(fft_amp**2) / n)

        # 计算频率标准差
        f_std = np.sqrt(np.sum((f - f_mean) ** 2 * fft_amp**2) / n)

        # 计算中心频率
        f_median = np.sum(f * fft_amp) / np.sum(fft_amp)

        # 构建频域特征向量
        frequency_domain_feature = [f_mean, f_max, f_rms, f_std, f_median]

        """===================时频域特征========================="""

        """-------------------------------离散小波变换---------------------------------------------"""

        # 将数据类型转换为张量
        time_domain_feature = torch.tensor(time_domain_feature)
        frequency_domain_feature = torch.tensor(frequency_domain_feature)
        # 将时域特征和频域特征拼接成一个特征向量
        features_vetor = torch.cat((time_domain_feature, frequency_domain_feature), 0)
        return features_vetor


# 数据加载器
def bearing_dataloader(bearing_path, batch_size, num_workers=8):
    bearing_dataset = Bearing_Dataset(bearing_path)
    # bearing_dataloader = DataLoader(bearing_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    # 划分训练集和测试集
    train_size = int(0.8 * len(bearing_dataset))
    test_size = len(bearing_dataset) - train_size
    split_size = [train_size, test_size]
    train_dataset, test_dataset = random_split(bearing_dataset, split_size)

    # 创建训练集加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # 创建测试集加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


# 轴承寿命预测模型
class Bearing_Predictor(nn.Module):
    # 初始化函数搭建transformer网络
    def __init__(
        self,
        feature_dim,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        use_decoder=False,
    ):
        super().__init__()
        # 调用父类的初始化函数

        self.model_type = "Transformer"
        self.src_mask = None
        # self.pos_encoder = None
        self.prenet = nn.Linear(feature_dim, feature_dim)
        # embedding层用于将输入的特征进行更深层次的抽象
        # self.input_embedding = nn.Embedding(feature_dim, embedding_dim=embedding_dim)
        # 编码层使用transformerencoderlayer，这是pytorch中的transformer的实现
        self.encoderlayer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            self.encoderlayer, num_layers=num_encoder_layers
        )
        # TODO: 使用transformer定义的解码层
        if use_decoder:
            self.decoderlayer = nn.TransformerDecoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                self.decoderlayer, num_layers=num_encoder_layers
            )
        else:
            # 使用全连接层作为解码层
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, 1),
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
        # input/src:(batch_size x seq_len x feature_dim)
        # src = self.input_embedding(src)
        out = self.prenet(src)
        # 交换维度 batch_size x seq_len x feature_dim -> seq_len x batch_size x feature_dim
        out = out.permute(1, 0, 2)
        output = self.encoder(out)
        # output = output.transpose(0, 1)
        # stats = output.mean(dim=1)
        # output: (seq_len x batch_size x 1)
        output = self.decoder(output)
        # return F.log_softmax(output, dim=-1)
        return output


# 模型前向传播
def model_forward(features, rul, model, criterion, device):
    # features, rul = batch
    # features = features.float()  # 将输入数据转换为Float类型
    # 特征为4x18的张量，将其转换为4x1x18的张量
    # features = features.view(features.size(0), 1, features.size(1))
    # print("一个批次中特征的维度：", features.shape)
    # 更改rul的形状为batch_sizex10x1
    # rul = rul.view(rul.size(0), 1, rul.size(1))
    # print(rul.shape)
    features = features.to(device)
    rul = rul.to(device)
    # output形状为batch_size x seq_len x 1
    output = model(features)
    # 将output形状住转换为batch_size x seq_len x 1
    # output = output.permute(1, 0, 2)
    rul = rul.permute(1, 0, 2)
    loss = criterion(output, rul)
    # 计算精度
    accuracy = 1 - ((output - rul).abs().float().mean())

    return loss, accuracy


# 绘制损失和精度曲线
def show_loss_accuracy(loss, accuracy):
    import matplotlib.pyplot as plt

    # 设置画布大小
    plt.figure(figsize=(12, 5))
    plt.plot(loss, label="Loss")
    plt.plot(accuracy, label="Accuracy")
    # 设置横坐标和纵坐标的标签
    plt.xlabel("epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    # plt.show()
    # 保存图片
    index = len(os.listdir("./train_info_pic/"))
    # 检查当前文件夹是否存在
    if not os.path.exists(f"./train_info_pic/run{index}"):
        os.makedirs(f"./train_info_pic/run{index}")
    plt.savefig(f"./train_info_pic/run{index}/loss_accuracy.png")
    plt.close()


# 评估模型
def evaluate_model(model, test_dataloader, criterion, device):
    model.eval()
    average_loss = []
    average_acc = []
    # eval_loss = 0
    # eval_accuracy = 0
    with torch.no_grad():
        for src, rul in test_dataloader:
            loss, accuracy = model_forward(src, rul, model, criterion, device)
            average_loss.append(loss.item())
            average_acc.append(accuracy.item())
    return sum(average_loss) / len(average_loss), sum(average_acc) / len(average_acc)


# 训练模型并评估
def train_model(
    model,
    train_dataloader,
    eval_dataloder,
    criterion,
    optimizer,
    device,
    total_steps,
    eval_steps,
    save_stpes,
    use_epoch=False,
    epochs=10,
):
    model.to(device)
    best_accuracy = 0.90
    best_state_dict = None
    total_loss = []
    total_accuracy = []
    if use_epoch:
        # 使用epoch进行训练
        for step in range(epochs):
            model.train()
            average_loss = []
            average_acc = []
            batch_index = 0
            for src, rul in train_dataloader:
                loss, accuracy = model_forward(src, rul, model, criterion, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                average_loss.append(loss.item())
                average_acc.append(accuracy.mean().item())
                # 每10个batch打印一个batch的损失和精度
                if batch_index % 10 == 0:
                    print(
                        "Epoch: %d, Batch num: %d, Loss: %.4f, Accuracy: %.4f"
                        % (step, batch_index, loss.item(), accuracy.mean().item())
                    )
                batch_index += 1
            total_loss.append(sum(average_loss) / len(average_loss))
            total_accuracy.append(sum(average_acc) / len(average_acc))
            print(
                "Epoch: %d, Loss: %.4f, Accuracy: %.4f"
                % (step, total_loss[-1], total_accuracy[-1])
            )
    else:
        # 使用total_steps进行训练
        train_iterator = iter(train_dataloader)
        for step in range(total_steps):
            model.train()
            average_loss = []
            average_acc = []
            try:
                src, rul = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                src, rul = next(train_iterator)
            # 前向传播
            loss, accuracy = model_forward(src, rul, model, criterion, device)
            batch_loss = loss.item()
            batch_accuracy = accuracy.item()
            # 反向传播
            loss.backward()
            # 更新
            optimizer.step()
            optimizer.zero_grad()
            # 记录损失和精度
            average_loss.append(loss.item())
            average_acc.append(accuracy.mean().item())
            total_loss.append(batch_loss)
            total_accuracy.append(batch_accuracy)
            # 每2步打印一次损失和精度
            if step % 2 == 0:
                print(
                    "Step num: %d, Loss: %.4f, Accuracy: %.4f"
                    % (step, batch_loss, batch_accuracy)
                )
            # 每100步打印最近100步的平均损失和精度
            if (step + 1) % 100 == 0:
                average_loss_val = sum(average_loss) / len(average_loss)
                average_acc_val = sum(average_acc) / len(average_acc)
                print(
                    "-----------------------------------------------------------------------------------------------------------------------"
                )
                print(
                    "The average loss and accuracy within the last one hundred steps :"
                )
                print(
                    "Step num: %d, Loss: %.4f, Accuracy: %.4f"
                    % (step, average_loss_val, average_acc_val)
                )
                print(
                    "-----------------------------------------------------------------------------------------------------------------------"
                )
                average_loss = []
                average_acc = []

            # 进行验证
            if step % eval_steps == 0:
                loss_val, accuracy_val = evaluate_model(
                    model, eval_dataloder, criterion, device
                )
                print(
                    "-----------------------------------------------------------------------------------------------------------------------"
                )
                print(
                    "Epoch: %d, Step num: %d, Eval loss: %.4f, Eval accuracy: %.4f"
                    % (step, step, loss_val, accuracy_val)
                )
                print(
                    "-----------------------------------------------------------------------------------------------------------------------"
                )
                if accuracy_val > best_accuracy:
                    best_accuracy = accuracy_val
                    best_state_dict = model.state_dict()

            # 每save_stpes保存一次模型
            if step + 1 % save_stpes == 0 and best_state_dict is not None:
                torch.save(model.state_dict(), "bearing_predictor.pth")
                print(
                    "-----------------------------------------------------------------------------------------------------------------------"
                )
                print(f"Model has been saved at step: {step}")
                print(
                    "-----------------------------------------------------------------------------------------------------------------------"
                )
        # 绘制损失和精度曲线
        show_loss_accuracy(total_loss, total_accuracy)
        # 保存损失和精度数据为json文件
        with open(
            f"./train_info_data/train_info{datetime.now().strftime('%Y-%m-%d_%H:%M')}.json",
            "w",
        ) as f:
            json.dump(
                {
                    "loss": total_loss,
                    "accuracy": total_accuracy,
                    "best_accuracy": best_accuracy,
                },
                f,
            )


def main():
    # 超参数
    batch_size = 32
    # embedding_dim = 256
    num_heads = 2
    num_encoder_layers = 6
    dim_feedforward = 256
    dropout = 0.1
    # activation = 'relu'
    # epochs = 10
    total_steps = 5000
    eval_steps = 100
    save_steps = 1000
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info]: Using device: {device}")

    # 数据集路径
    bearing_path = "phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1"

    # 加载数据集
    train_bearing_dataloader, eval_bearing_dataloder = bearing_dataloader(
        bearing_path, batch_size
    )

    # 模型初始化
    model = Bearing_Predictor(
        feature_dim=18,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    # 损失函数
    criterion = nn.MSELoss()

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 训练模型并评估
    train_model(
        model,
        train_bearing_dataloader,
        eval_bearing_dataloder,
        criterion,
        optimizer,
        device,
        total_steps,
        eval_steps,
        save_steps,
        use_epoch=False,
    )

    # 保存模型
    torch.save(model, "bearing_predictor.pth")


if __name__ == "__main__":
    main()
