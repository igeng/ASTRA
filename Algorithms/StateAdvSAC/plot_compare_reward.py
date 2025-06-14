#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : ASTRA
@File    : plot_compare_reward.py
@Author  : igeng
@Date    : 2024/11/18 18:06 
@Descrip : Compare rewards from different algorithms
'''

import pandas as pd
import matplotlib.pyplot as plt

# 设置默认字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 读取CSV文件
# DQN
file_path1 = 'results_hyatssm_DQN.csv'  # 替换为你的CSV文件路径
data1 = pd.read_csv(file_path1, header=None)[:200]
# 提取episode和reward数据
episodes1 = data1.iloc[:, 0]  # 第一列作为episode
rewards1 = data1.iloc[:, 3]   # 第四列作为reward

# Dueling DQN
file_path2 = 'results_hyatssm_dudqn_256_128.csv'  # 替换为你的CSV文件路径
data2 = pd.read_csv(file_path2, header=None)[:200]
# 提取episode和reward数据
episodes2 = data2.iloc[:, 0]  # 第一列作为episode
rewards2 = data2.iloc[:, 3]   # 第四列作为reward

# PPO
file_path3 = 'results_hyatssm_ppo.csv'  # 替换为你的CSV文件路径
data3 = pd.read_csv(file_path3, header=None)[:200]
# 提取episode和reward数据
episodes3 = data3.iloc[:, 0]  # 第一列作为episode
rewards3 = data3.iloc[:, 3]   # 第四列作为reward

# SAC
file_path4 = 'results_hyatssm_sac.csv'  # 替换为你的CSV文件路径
data4 = pd.read_csv(file_path4, header=None)[:200]
# 提取episode和reward数据
episodes4 = data4.iloc[:, 0]  # 第一列作为episode
rewards4 = data4.iloc[:, 3]   # 第四列作为reward

# AdvSAC
file_path5 = 'results_hyatssm_stt_adv_sac.csv'  # 替换为你的CSV文件路径
data5 = pd.read_csv(file_path5, header=None)[:200]
# 提取episode和reward数据
episodes5 = data5.iloc[:, 0]  # 第一列作为episode
rewards5 = data5.iloc[:, 3]   # 第四列作为reward

# 绘制图形
plt.figure(figsize=(12, 8))

# 绘制AdvSAC的奖励曲线
plt.plot(episodes5, rewards5, label='ASTRA', marker='o', linestyle='-', markersize=4)

# 绘制SAC的奖励曲线
plt.plot(episodes4, rewards4, label='SAC', marker='^', linestyle='-', markersize=4)

# 绘制DQN的奖励曲线
plt.plot(episodes1, rewards1, label='DQN', marker='o', linestyle='-', markersize=4)

# 绘制Dueling DQN的奖励曲线
plt.plot(episodes2, rewards2, label='Dueling DQN', marker='x', linestyle='-', markersize=4)

# 绘制PPO的奖励曲线
plt.plot(episodes3, rewards3, label='PPO', marker='s', linestyle='-', markersize=4)


# 设置图表标题和标签
# plt.title('Comparison of Rewards from Different Algorithms', fontsize=24)
plt.xlabel('Episodes', fontsize=20)
plt.ylabel('Rewards', fontsize=20)

# 显示图例
plt.legend(fontsize=20)

# 显示网格
plt.grid(True)

# 设置坐标轴刻度标签的字体大小
plt.tick_params(axis='both', which='major', labelsize=16)

# 保存图形为SVG格式，设置dpi为900，并去除多余空白部分
plt.savefig('reward_comparison.svg', format='svg', dpi=900, bbox_inches='tight')

# 显示图形
plt.show()