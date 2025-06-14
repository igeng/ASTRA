#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : PerformanceEstimatorModel 
@File    : plot_req.py
@Author  : igeng
@Date    : 2024/12/6 11:10 
@Descrip :
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置默认字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 读取文件
file_path = 'nasa-900.req'

# 使用 pandas 读取文件，假设文件中只有一列数据
data = pd.read_csv(file_path, header=None, names=['concurrent_users'])

# 添加时间索引（假设每行代表一个单位时间）
data['time'] = range(1, len(data) + 1)

# 设置时间为索引（可选，方便绘图时使用）
data.set_index('time', inplace=True)

# 绘制图表
plt.figure(figsize=(12, 4))

# 使用 seaborn 绘制折线图，并为折线图添加标签
sns.lineplot(data=data, x='time', y='concurrent_users', label='Concurrent Users')

# 添加标题和标签
# plt.title('Configured Concurrent Users Over the Experiment', fontsize=24)
plt.xlabel('Time (5s)', fontsize=20)
plt.ylabel('Number of Concurrent Users', fontsize=20)

# 显示网格
plt.grid(True)

# 设置坐标轴刻度标签的字体大小
plt.tick_params(axis='both', which='major', labelsize=16)

# 添加图例，并设置图例的字体大小
plt.legend(fontsize=16)

# 保存图表（可选）
plt.savefig('concurrent_users_over_time_nasa.svg', format='svg', dpi=900, bbox_inches='tight')
plt.savefig('concurrent_users_over_time_nasa.pdf', format='pdf', dpi=900, bbox_inches='tight')

# 显示图表
plt.show()