#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : ASTRA
@File    : HyAtSSM_TD_DRL_load.py
@Author  : igeng
@Date    : 2024/10/31 16:43
@Descrip : 学习的初始化参数（Learnable Initialization）
这是状态空间模型中非常推荐的初始化方法。
在模型中定义隐状态初始值为可学习参数，允许模型通过训练自动调整隐状态初值。
这种方法使得隐状态可以根据数据分布进行优化，更加适应具体任务的要求。
这里是标准的SSM模型构建
增加新的 mae loss
加上load后缀就是在全部数据集上进行训练
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os
import torch.optim.lr_scheduler as lr_scheduler

class AttentionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_features):
        super(AttentionModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_features = num_features
        
        # 时间维度注意力
        self.temporal_query = nn.Linear(input_dim, hidden_dim * num_heads * self.num_features)
        self.temporal_key = nn.Linear(input_dim, hidden_dim * num_heads * self.num_features)
        self.temporal_value = nn.Linear(input_dim, hidden_dim * num_heads * self.num_features)
        
        # 特征间注意力(类似SE模块)
        self.fc_squeeze = nn.Linear(hidden_dim * num_heads * self.num_features, (hidden_dim * num_heads) // 2)
        self.layer_norm_squeeze = nn.LayerNorm((hidden_dim * num_heads) // 2)
        self.fc_excitation = nn.Linear((hidden_dim * num_heads) // 2, hidden_dim * num_heads * self.num_features)
        self.layer_norm_excitation = nn.LayerNorm(hidden_dim * num_heads * self.num_features)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(hidden_dim * num_heads, input_dim)
        self.layer_norm_out = nn.LayerNorm(input_dim)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
    
        # 时间维度注意力
        temporal_queries = self.temporal_query(x).view(batch_size, seq_len, num_features, self.num_heads, self.hidden_dim).transpose(2, 3)
        temporal_keys = self.temporal_key(x).view(batch_size, seq_len, num_features, self.num_heads, self.hidden_dim).transpose(2, 3)
        temporal_values = self.temporal_value(x).view(batch_size, seq_len, num_features, self.num_heads, self.hidden_dim).transpose(2, 3)
        # "bqchd": batch_size, query, input_dim, head, hidden_dim
        temporal_energy = torch.einsum("bqchd,bkchd->bhqck", [temporal_queries, temporal_keys])
        temporal_attention = torch.softmax(temporal_energy / (self.hidden_dim ** 0.5), dim=-1)
        temporal_out = torch.einsum("bhqck,bkchd->bqchd", [temporal_attention, temporal_values]).transpose(2, 3).reshape(batch_size, seq_len, num_features, -1)
        
        # 特征间注意力(类似SE模块)
        spatial_squeeze = torch.mean(temporal_out, dim=1)
        spatial_squeeze = spatial_squeeze.view(batch_size, -1)
        spatial_squeeze = self.layer_norm_squeeze(self.fc_squeeze(spatial_squeeze))
        spatial_excitation = self.layer_norm_excitation(self.fc_excitation(spatial_squeeze))
        spatial_excitation = self.sigmoid(spatial_excitation).view(batch_size, 1, num_features, -1)
        spatial_out = temporal_out * spatial_excitation
        
        out = self.layer_norm_out(self.fc(spatial_out))

        return out

class StateSpaceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, variational=False):
        super(StateSpaceModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.variational = variational
        
        observation_layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            observation_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
        observation_layers.append(nn.Linear(hidden_dim, output_dim))
        self.observation = nn.Sequential(*observation_layers)

        self.combine = nn.Linear(input_dim*2, input_dim)
        self.transition = nn.Linear(output_dim * output_dim, hidden_dim)
        self.layer_norm_transition = nn.LayerNorm(hidden_dim)

        # 在 transition 和 output_layer 之间加入 GELU 激活层
        self.gelu = nn.GELU()

        # 线性层将 hidden_dim 转换为 output_dim * output_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim * output_dim)
        # states = [torch.zeros(batch_data.size(0), hidden_dim).to(device) for _ in range(num_ssm_modules)]
        # `attended_x` shape: [batch_size, seq_len, num_features, hidden_dim * num_heads]
        # self.state_init = nn.Parameter(torch.zeros(1, hidden_dim))
        # self.state_init = nn.Parameter(torch.zeros(1, seq_len, num_features, output_dim))
        self.state_init = nn.Parameter(torch.zeros(1, 1, 8, output_dim))

        if self.variational:
            self.mean = nn.Parameter(torch.zeros(hidden_dim))
            self.log_var = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        # x: t=1 shape:torch.Size([128, 1, 8, 8]) [batch_size, seq_len, ]
        batch_size = x.size(0)
        # 初始化状态 state_0
        state_pre = self.state_init.repeat(batch_size, 1, 1, 1)
        # observation = self.observation(x)
        # 状态转移方程
        combined_tensor = torch.cat((state_pre, x), dim=-1)
        state = self.combine(combined_tensor.view(-1, self.output_dim*2)).view(batch_size, 1, self.output_dim, self.output_dim)

        state_transformed = self.transition(state.view(state.size(0), state.size(1), -1))

        # 应用 transition 和 GELU 激活层
        state_transformed = self.gelu(state_transformed)

        state_normalized = self.layer_norm_transition(state_transformed)
        if self.variational:
            # state_1
            state = state_normalized + torch.randn_like(state_normalized) * torch.exp(self.log_var).unsqueeze(0).unsqueeze(1) + self.mean.unsqueeze(0).unsqueeze(1)
        else:
            state = state_normalized
        state = self.output_layer(state)
        # 这里得到t时刻的中间状态
        state = state.view(state_pre.size(0), state_pre.size(1), self.output_dim, self.output_dim)

        # observation = self.observation(x)
        observation = self.observation(state)
        return observation, state


class AtSSMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_ssm_modules, num_ssm_layers, num_features, variational=False):
        super(AtSSMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_ssm_modules = num_ssm_modules
        self.num_features = num_features

        # 初始化 Attention 和 StateSpace 模块
        self.attention = AttentionModule(input_dim, hidden_dim, num_heads, num_features)
        self.ssm_modules = nn.ModuleList([StateSpaceModel(input_dim, hidden_dim, output_dim, num_ssm_layers, variational=variational) for _ in range(num_ssm_modules)])

        # 通过全连接层来合并 num_features 维度信息
        self.merge_layer = nn.Linear(num_features * output_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_features, input_dim]
        attended_x = self.attention(x)
        # `attended_x` shape: [batch_size, seq_len, num_features, input_dim]

        observations = []
        new_states = []

        for ssm_module in self.ssm_modules:
            observation, new_state = ssm_module(attended_x)
            observations.append(observation)
            new_states.append(new_state)

        # 将所有 ssm_modules 的输出进行平均
        observation = torch.stack(observations).mean(dim=0)
        # `observation` shape: [batch_size, seq_len, num_features, output_dim]

        # 重新排列维度并通过全连接层合并 num_features 信息
        observation = observation.view(observation.size(0), observation.size(1), -1)  # 展平成 [batch_size, seq_len, num_features * output_dim]
        observation = self.merge_layer(observation)  # 应用全连接层得到 [batch_size, seq_len, output_dim]

        return observation, new_states

class CustomLoss(nn.Module):
    def __init__(self, weight_mse=0.7, weight_mae=0.3):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()  # 或者使用 SmoothL1Loss()
        self.weight_mse = weight_mse
        self.weight_mae = weight_mae

    def forward(self, predictions, targets):
        loss_mse = self.mse(predictions, targets)
        loss_mae = self.mae(predictions, targets)
        # 组合损失函数
        combined_loss = self.weight_mse * loss_mse + self.weight_mae * loss_mae
        return combined_loss


def calculate_metrics(predicted, actual, service_index, service_name):
    mae = mean_absolute_error(actual[:, service_index], predicted[:, service_index])
    mse = mean_squared_error(actual[:, service_index], predicted[:, service_index])
    rmse = np.sqrt(mse)

    print(f"{service_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")


# Function to create time series data with a sequence length of 4
def create_sequences(data, seq_length=4):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# 绘图函数
def plot_figure_compare(workload, actual, wl_name, model_dir):
    plt.figure(figsize=(10, 5))
    # 绘制预测值
    plt.plot(workload, color='cyan', label='Predicted ' + wl_name)
    # 绘制真实值
    plt.plot(actual, color='orange', label='Actual ' + wl_name, linestyle='--')
    # 设置标题
    plt.title('Microservice Performance Indicator: ' + wl_name, fontsize=16, fontweight='bold', color='black')
    # 设置坐标轴标签
    plt.xlabel('Time', fontsize=14, color='black')
    plt.ylabel(wl_name, fontsize=14, color='black')
    # 设置图例
    plt.legend(loc='upper left', fontsize=12)
    # 设置网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    # 设置背景颜色
    plt.gca().set_facecolor('white')
    # 保存图形
    # file_path = os.path.join(model_dir, f'{wl_name}_comparison.png')
    plt.savefig(model_dir, bbox_inches='tight', dpi=300)
    # 显示图形
    plt.show()

if __name__ == '__main__':
    # 创建模型保存的文件夹
    model_dir = 'HyAtSSM_TD_DRL_load'
    os.makedirs(model_dir, exist_ok=True)  # 确保文件夹存在

    services = ['carts', 'catalogue', 'front-end', 'orders', 'payment', 'shipping', 'user']

    # 1. 加载预处理数据
    df = pd.read_csv('HyAtSSM_TD_DRL.csv')

    # 定义输入和输出列
    input_columns = ['total_request', 'carts_num_pods', 'catalogue_num_pods', 'front-end_num_pods',
                     'orders_num_pods', 'payment_num_pods', 'shipping_num_pods', 'user_num_pods']
    output_columns = ['carts_latency', 'carts_cpu_usage', 'catalogue_cpu_usage', 'front-end_cpu_usage',
                      'orders_cpu_usage', 'payment_cpu_usage', 'shipping_cpu_usage', 'user_cpu_usage']

    # 数据预处理
    X = df[input_columns]
    y = df[output_columns]

    X_scaled = X.to_numpy()
    y_scaled = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False)

    # 在所有的数据上进行训练
    X_train = np.vstack((X_train, X_test))
    y_train = np.vstack((y_train, y_test))

    # 转换为 PyTorch 张量并添加序列维度
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # [batch_size, seq_len, input_size]
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    BATCH_SIZE = 128
    # 数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create DataLoader for the test dataset
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 设置模型超参数
    input_dim = X_train_tensor.size(-1) # 8
    hidden_dim = 128
    output_dim = y_train_tensor.size(-1) # 8
    num_heads = 2
    num_ssm_modules = 4
    num_ssm_layers = 4

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # 初始化模型
    model = AtSSMModel(input_dim, hidden_dim, output_dim, num_heads, num_ssm_modules, num_ssm_layers, len(input_columns), variational=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # criterion = nn.MSELoss()
    # 初始化自定义损失函数
    criterion = CustomLoss(weight_mse=0.4, weight_mae=0.6)  # 调整权重

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 训练模型
    num_epochs = 1000
    best_loss = float('inf')  # 初始化最优损失为无穷大
    best_model_path = os.path.join(model_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_data, batch_targets in train_loader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            optimizer.zero_grad()

            # 移除 states，并直接传入 batch_data
            preds, _ = model(batch_data)
            loss = criterion(preds.view(batch_targets.size()), batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_data.size(0)

        # 更新学习率
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')

        # 检查是否为最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with loss: {best_loss:.4f}')

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    print('Best model loaded from', best_model_path)

    # 评估模型
    model.eval()
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for batch_data, batch_targets in test_loader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)

            # 移除 states 并直接传入 batch_data
            preds, _ = model(batch_data)
            preds = preds.squeeze(1)

            y_pred_list.extend(preds.cpu().numpy())
            y_true_list.extend(batch_targets.cpu().numpy())

    # 汇总预测和真实值
    all_predictions = np.vstack(y_pred_list)
    all_targets = np.vstack(y_true_list)

    predicted, actual = all_predictions, all_targets
    result_dir = f"HyAtSSM_TD_DRL_load/"
    os.makedirs(result_dir, exist_ok=True)  # 确保文件夹存在
    for i in range(len(output_columns)):
        fg_name = result_dir+output_columns[i]+".png"
        plot_figure_compare(predicted[:,i], actual[:,i], output_columns[i], fg_name)


    # 收集所有服务的性能指标
    results = []
    # 计算 latency 的评价指标
    calculate_metrics(predicted, actual, service_index=0, service_name='carts_latency')
    results.append(('carts_latency', mean_absolute_error(actual[:, 0], predicted[:, 0]),
                    mean_squared_error(actual[:, 0], predicted[:, 0]),
                    np.sqrt(mean_squared_error(actual[:, 0], predicted[:, 0]))))

    # 计算 CPU 使用率的评价指标
    for i, service in enumerate(services, 1):
        calculate_metrics(predicted, actual, service_index=i, service_name=f'{service}_cpu_usage')
        results.append((f'{service}_cpu_usage', mean_absolute_error(actual[:, i], predicted[:, i]),
                        mean_squared_error(actual[:, i], predicted[:, i]),
                        np.sqrt(mean_squared_error(actual[:, i], predicted[:, i]))))

    # 将性能指标保存到CSV文件
    performance_df = pd.DataFrame(results, columns=['Service', 'MAE', 'MSE', 'RMSE'])
    performance_csv_path = os.path.join(model_dir, 'performance_metrics.csv')
    # performance_csv_path = os.path.join(model_dir, str(config.n_layers), 'performance_metrics.csv')
    performance_df.to_csv(performance_csv_path, index=False)

    print(f"Performance metrics have been saved to {performance_csv_path}")

    # Initialize the total number of parameters
    total_params = 0

    # Iterate through the model parameters
    for param in model.parameters():
        # Increment the total number of parameters
        total_params += param.numel()

    # Print the total number of parameters
    print("Number of parameters in Mamba model:", total_params)
