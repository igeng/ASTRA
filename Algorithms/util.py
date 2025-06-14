#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : PerformanceEstimatorModel 
@File    : util.py
@Author  : igeng
@Date    : 2024/10/11 14:03 
@Descrip :
'''
import csv

def save_to_csv_lg(file_name, episode, avg_pods, avg_latency, avg_cpu_util, reward, execution_time):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='')
    with file:
        fields = ['episode', 'avg_pods',
                  'avg_latency',
                  'avg_cpu_util',
                  'reward', 'execution_time']
        writer = csv.DictWriter(file, fieldnames=fields)
        # writer.writeheader()
        writer.writerow(
            {'episode': episode,
             'avg_pods': float("{:.2f}".format(avg_pods)),
             'avg_latency': float("{:.4f}".format(avg_latency)),
             'avg_cpu_util': float("{:.4f}".format(avg_cpu_util)),
             'reward': float("{:.2f}".format(reward)),
             'execution_time': float("{:.2f}".format(execution_time))}
        )

import csv
import os

def save_to_csv(file_name, episode, avg_pods, avg_latency, reward, execution_time):
    # 打开文件，使用 'a+' 模式进行追加和读取
    with open(file_name, 'a+', newline='') as file:
        # 移动指针到文件开头，检查文件是否为空
        file.seek(0)
        is_file_empty = file.read(1) == ''

        # 定义字段名
        fields = ['episode', 'avg_pods', 'avg_latency', 'reward', 'execution_time']
        writer = csv.DictWriter(file, fieldnames=fields)

        # 如果文件为空，写入表头
        if is_file_empty:
            writer.writeheader()

        # 写入数据行
        writer.writerow({
            'episode': episode,
            'avg_pods': float("{:.2f}".format(avg_pods)),
            'avg_latency': float("{:.4f}".format(avg_latency)),
            'reward': float("{:.2f}".format(reward)),
            'execution_time': float("{:.2f}".format(execution_time))
        })

def save_to_csv_eval(file_name, episode, avg_pods, avg_latency, slo_vlt, avg_cpu_util, reward, execution_time):
    # 打开文件，使用 'a+' 模式进行追加和读取
    with open(file_name, 'a+', newline='') as file:
        # 移动指针到文件开头，检查文件是否为空
        file.seek(0)
        is_file_empty = file.read(1) == ''

        # 定义字段名
        fields = ['episode', 'avg_pods', 'avg_latency', 'slo_violation', 'avg_cpu_util', 'reward', 'execution_time']
        writer = csv.DictWriter(file, fieldnames=fields)

        # 如果文件为空，写入表头
        if is_file_empty:
            writer.writeheader()

        # 写入数据行
        writer.writerow({
            'episode': episode,
            'avg_pods': float("{:.2f}".format(avg_pods)),
            'avg_latency': float("{:.4f}".format(avg_latency)),
            'slo_violation': float("{:.4f}".format(slo_vlt)),
            'avg_cpu_util': float("{:.4f}".format(avg_cpu_util)),
            'reward': float("{:.2f}".format(reward)),
            'execution_time': float("{:.2f}".format(execution_time))
        })