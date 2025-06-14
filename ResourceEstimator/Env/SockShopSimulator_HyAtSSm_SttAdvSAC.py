#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : ASTRA
@File    : SockShopSimulator_HyAtSSm_SttAdvSAC.py
@Author  : igeng
@Date    : 2024/9/25 15:53
@Descrip : Build simulation environment based on trained HyAtSSM.
'''
import logging
import time
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import os
import numpy as np
from Algorithms.util import save_to_csv
import torch

from ResourceEstimator.HyAtSSM.HyAtSSM_TD_DRL_load import AtSSMModel

MIN_RPS = 0.0
# MIN and MAX Replication
MIN_REPLICATION = 1
MAX_REPLICATION = 8
MAX_RPS = 10000
MAX_CPU = 10000
MAX_RT = 10000
MAX_RCR = 10000
MAX_SLAVR = 10000

MAX_STEPS = 10000  # MAX Number of steps per episode

a = [0,0]

class SockShopSimulatorHyAtSSMSttAdvSAC(gym.Env):
    """AutoScaling for Sock Shop in Kubernetes - an OpenAI gym environment"""

    metadata = {'render.modes': ['ansi', 'array']}

    def __init__(self, start_step=0, train_step=10000, sla_threshold=0.75):
        # Define action and observation space. They must be gym.spaces objects
        super(SockShopSimulatorHyAtSSMSttAdvSAC, self).__init__()
        self.name = "sock_shop_simulator_gym"
        self.__version__ = "0.1"
        self.seed()

        logging.info("[Init] Env: {} | Version {} |".format(self.name, self.__version__))

        self.start_step = start_step

        # 用于指示当前的时间步
        self.current_step = 0

        self.pod_num = [1, 1, 1, 1, 1, 1, 1]

        self.train_step = train_step

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 3

        self.action_space = spaces.MultiDiscrete([7, self.num_actions])

        # Observations: 6 Metrics! -> 1 + 7 * 5 = 36
        # "rps"                  -> request per second
        # +
        # "instance_replicate"   -> Number of deployed Pods
        # "cpu_util"             ->
        # "response_time"        ->
        # "request_cr"           -> Request change rate
        # "SLA_vr"               -> SLA violation rate

        self.min_rps = MIN_RPS
        self.min_pods = MIN_REPLICATION
        self.max_pods = MAX_REPLICATION

        self.observation_space = self.get_observation_space()

        # Info
        self.total_reward = None

        self.avg_latency = []

        # episode over
        self.episode_over = False
        self.info = {}
        self.episode_over_state = []

        # Keywords for Reward calculation
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False
        self.cost_weight = 0  # add here a value to consider cost in the reward function

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results_hyatssm_stt_adv_sac.csv"
        self.obs_csv = self.name + "_observation_hyatssm_stt_adv_sac.csv"

        # 定义服务名称列表
        self.services = [
            'carts', 'catalogue', 'front-end', 'orders', 'payment', 'shipping', 'user'
        ]

        # 目标变量模板
        self.targets_template = [
            '_latency', '_cpu_usage'
        ]
        root_dir = f'ASTRA'

        # 创建模型加载的文件夹路径
        self.model_dir = root_dir+'/ResourceEstimator/HyAtSSM/HyAtSSM_TD_DRL_load/'
        # 读取测试数据
        self.rps_data_path = root_dir+'/ResourceEstimator/HyAtSSM/HyAtSSM_TD_DRL.csv'
        self.rps_data = pd.read_csv(self.rps_data_path, delimiter=',')
        # 构建特征和目标变量列表，从csv文件里读取的只有rps
        features = ['total_request']

        # 读取rps数据
        # 这里还应该读取每个服务的pod个数，即特征列：*_num_pods列
        self.train_rps_data = self.rps_data[features]

        # 初始化模型存储
        self.models = None
        self.best_model_path = os.path.join(self.model_dir, 'best_model.pth')
        # 初始化模型存储
        self._load_models()

        self.done = False

        self.sla_threshold = sla_threshold
        self.avg_res = 0
        self.avg_cpu_util = 0
        self.res, self.cpu_util = [], []
        self.current_req = 0


    def _load_models(self):
        # 定义输入和输出列
        input_columns = ['total_request', 'carts_num_pods', 'catalogue_num_pods', 'front-end_num_pods',
                         'orders_num_pods', 'payment_num_pods', 'shipping_num_pods', 'user_num_pods']
        output_columns = ['carts_latency', 'carts_cpu_usage', 'catalogue_cpu_usage', 'front-end_cpu_usage',
                          'orders_cpu_usage', 'payment_cpu_usage', 'shipping_cpu_usage', 'user_cpu_usage']

        # 设置模型超参数
        input_dim = 8
        hidden_dim = 128
        output_dim = 8
        num_heads = 2
        num_ssm_modules = 4
        num_ssm_layers = 4
        model = AtSSMModel(input_dim, hidden_dim, output_dim, num_heads, num_ssm_modules, num_ssm_layers,
                           len(input_columns), variational=True)
        model.load_state_dict(torch.load(self.best_model_path))
        print('Best model loaded from', self.best_model_path)
        self.models = model

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.

        一开始所有服务的 pod 个数都是1，但是在每个人为划分的episode结束后，即每个episode的step走完以后，
        应该是从上一个episode最后的状态开始新的episode
        """
        # 重置用户请求信息，即当前的时间步重置为start_step
        self.current_step = self.start_step
        self.pod_num = [1, 1, 1, 1, 1, 1, 1]
        self.episode_over = False
        self.done = False
        self.total_reward = 0

        self.avg_latency = []

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        self.episode_over_state = self.get_state(self.current_step, self.pod_num)

        return np.array(self.episode_over_state)

    def render(self, mode='array', close=False):
        # Render the environment to the screen
        return

    # revision here!
    def step(self, action):
        # print("Before trans, action is : {}".format(action))

        action = self.trans(action)

        # print("After trans, action is : {}".format(action))

        if self.current_step == 0:
            self.time_start = time.time()

        # 根据action更新微服务pod数量的列表
        self.pod_num[action[0]] += action[1]

        illegal_reward_penalty = 0
        if self.pod_num[action[0]] < 1:
            illegal_reward_penalty = -3
        elif self.pod_num[action[0]] > 8:
            illegal_reward_penalty = -3

        self.episode_over = False

        # 使用列表推导式处理列表
        pod_num_new = [min(8, max(1, num)) for num in self.pod_num]
        # print("The new pod is: {}".format(pod_num_new))

        self.pod_num = pod_num_new

        self.current_step += 1

        # 返回新的状态
        next_state = self.get_state(self.current_step, self.pod_num)
        # print(next_state)

        # 返回即时奖励
        reward = self.get_reward(self.avg_res, self.avg_cpu_util) + illegal_reward_penalty
        # print("Current epiosde {}, step {}, reward is {}".format(self.episode_count, self.current_step, reward))

        self.total_reward += reward

        if self.current_step >= self.train_step - 1:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            self.episode_over_state = next_state
            # done
            self.episode_over = True
            save_to_csv(self.file_results, self.episode_count, 1, 1,
                        self.total_reward, self.execution_time)

        return np.array(next_state), reward, self.episode_over, self.info



    def trans(self, action):
        # 定义一个字典来映射动作到服务索引和服务变更
        # 每个服务的变更范围是[-1,1]
        mapping = {
            0: (0, -1),  # carts, decrease by 1
            1: (0, 0),  # carts, no change
            2: (0, 1),  # carts, increase by 1

            3: (1, -1),  # catalogue, decrease by 1
            4: (1, 0),  # catalogue, no change
            5: (1, 1),  # catalogue, increase by 1

            6: (2, -1),  # front-end, decrease by 1
            7: (2, 0),  # front-end, no change
            8: (2, 1),  # front-end, increase by 1

            9: (3, -1),  # orders, decrease by 1
            10: (3, 0),  # orders, no change
            11: (3, 1),  # orders, increase by 1

            12: (4, -1),  # payment, decrease by 1
            13: (4, 0),  # payment, no change
            14: (4, 1),  # payment, increase by 1

            15: (5, -1),  # shipping, decrease by 1
            16: (5, 0),  # shipping, no change
            17: (5, 1),  # shipping, increase by 1

            18: (6, -1),  # user, decrease by 1
            19: (6, 0),  # user, no change
            20: (6, 1),  # user, increase by 1
        }

        # 获取对应的服务索引和服务变更
        service_index, change = mapping[action]
        return service_index, change

    def get_latest_rps_by_step(self, index_step):

        latest_rps = self.train_rps_data.iloc[index_step]

        return latest_rps.item()

    def get_state(self, current_step, pod_num):

        req = self.get_latest_rps_by_step(current_step)
        self.current_req = req

        if current_step == 0:
            pre_req = req
        else:
            pre_req = self.get_latest_rps_by_step(current_step-1)

        #########################
        # 应对rps为0的情况，方法1
        if pre_req == 0:
            pre_req = 0.1
            # print("The pre_req of {} step is 0".format(current_step))
        ##########################

        state = []
        self.res, self.cpu_util = [], []

        input = [req]
        input.extend(pod_num)
        # 将列表转换为张量
        input_tensor = torch.tensor(input)
        # 调整张量的形状为 [1, 1, 8]
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        preds, _ = self.models(input_tensor)
        preds = preds.squeeze(1).squeeze(1)
        # preds[0][0].item() float

        # 对每个服务加载模型并进行预测
        for i in range(len(self.services)):
            # 请求率、实例数
            state.append(req)
            state.append(pod_num[i])
            for target in self.targets_template:
                # 这里的action应该是根据上一个timestep的pod，加上修改的action，最后得到新的pod个数
                if self.models is not None:
                    # 使用模型进行预测
                    # 预测分别是 响应时间 和 CPU利用率
                    if target == '_latency':
                        y_pred = preds[0][0]
                        # 响应时间
                        state.append(y_pred.item())
                        self.res.append(y_pred)
                    else:
                        y_pred = preds[0][i+1]
                        # CPU使用率
                        state.append(y_pred.item())
                        self.cpu_util.append(y_pred)

            # 应对rps为0的情况，方法2
            # state.append((req - pre_req) / (pre_req+0.1))

            # 请求变化率和SLA违反状态
            state.append((req - pre_req)/pre_req)
            state.append((self.res[-1] / self.sla_threshold).item())

        # 这两行代码随着step增加，计算量会增加
        self.avg_res = (sum(self.res) / len(self.res)).item()
        self.avg_cpu_util = (sum(self.cpu_util) / len(self.cpu_util)).item()
        # print(state)
        # 状态组成包括每个服务的请求率、实例数、CPU使用率、响应时间、"请求变化率和 SLA 违反状态"
        return state

    def get_reward(self, avg_res, avg_cpu_util):

        if avg_res <= self.sla_threshold:
            reward = avg_cpu_util
        else:
            reward = (-avg_res / self.sla_threshold)+avg_cpu_util

        return reward

    def get_observation_space(self):
            return spaces.Box(
                low=np.array([
                    self.min_rps,
                    self.min_pods,  # Number of Pods  -- 1) carts
                    0,  # CPU Util
                    0,  # Response time
                    0,  # Request change rate
                    0,  # SLA violation rate
                    self.min_rps,
                    self.min_pods,  # Number of Pods -- 2) catalogue
                    0,  # CPU Util
                    0,  # Response time
                    0,  # Request change rate
                    0,  # SLA violation rate
                    self.min_rps,
                    self.min_pods,  # Number of Pods -- 3) front-end
                    0,  # CPU Util
                    0,  # Response time
                    0,  # Request change rate
                    0,  # SLA violation rate
                    self.min_rps,
                    self.min_pods,  # Number of Pods -- 4) orders
                    0,  # CPU Util
                    0,  # Response time
                    0,  # Request change rate
                    0,  # SLA violation rate
                    self.min_rps,
                    self.min_pods,  # Number of Pods -- 5) payment
                    0,  # CPU Util
                    0,  # Response time
                    0,  # Request change rate
                    0,  # SLA violation rate
                    self.min_rps,
                    self.min_pods,  # Number of Pods -- 6) shipping
                    0,  # CPU Util
                    0,  # Response time
                    0,  # Request change rate
                    0,  # SLA violation rate
                    self.min_rps,
                    self.min_pods,  # Number of Pods -- 7) user
                    0,  # CPU Util
                    0,  # Response time
                    0,  # Request change rate
                    0,  # SLA violation rate
                ]), high=np.array([
                    MAX_RPS,
                    self.max_pods,  # Number of Pods -- 1)
                    MAX_CPU,  # CPU Usage (in m)
                    MAX_RT,
                    MAX_RCR,
                    MAX_SLAVR,
                    MAX_RPS,
                    self.max_pods,  # Number of Pods -- 2)
                    MAX_CPU,  # CPU Usage (in m)
                    MAX_RT,
                    MAX_RCR,
                    MAX_SLAVR,
                    MAX_RPS,
                    self.max_pods,  # Number of Pods -- 3)
                    MAX_CPU,  # CPU Usage (in m)
                    MAX_RT,
                    MAX_RCR,
                    MAX_SLAVR,
                    MAX_RPS,
                    self.max_pods,  # Number of Pods -- 4)
                    MAX_CPU,  # CPU Usage (in m)
                    MAX_RT,
                    MAX_RCR,
                    MAX_SLAVR,
                    MAX_RPS,
                    self.max_pods,  # Number of Pods -- 5)
                    MAX_CPU,  # CPU Usage (in m)
                    MAX_RT,
                    MAX_RCR,
                    MAX_SLAVR,
                    MAX_RPS,
                    self.max_pods,  # Number of Pods -- 6)
                    MAX_CPU,  # CPU Usage (in m)
                    MAX_RT,
                    MAX_RCR,
                    MAX_SLAVR,
                    MAX_RPS,
                    self.max_pods,  # Number of Pods -- 7)
                    MAX_CPU,  # CPU Usage (in m)
                    MAX_RT,
                    MAX_RCR,
                    MAX_SLAVR,
                ]),
                dtype=np.float32
            )

if __name__ == "__main__":
    env = SockShopSimulatorHyAtSSMSttAdvSAC()
    env.reset()
    env.step(1)