#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : ASTRA 
@File    : train.py
@Author  : igeng
@Date    : 2024/11/15 5:36
@Descrip :
'''
from tqdm import tqdm
import numpy as np
from collections import deque
import torch
import argparse
from buffer import ReplayBuffer
from utils import save, collect_random
import random
from agent import SAC
from ResourceEstimator.Env.SockShopSimulator_HyAtSSm_SttAdvSAC import SockShopSimulatorHyAtSSMSttAdvSAC


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="StateAdvSAC", help="Run name, default: StateAdvSAC")
    parser.add_argument("--env", type=str, default="SockShopSimulatorHyAtSSMSttAdvSAC",
                        help="Default: SockShopSimulatorHyAtSSMSttAdvSAC")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100000,
                        help="Maximal training dataset size, default: 100000")
    # parser.add_argument("--seed", type=int, default=3407, help="Seed, default: 1")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=1, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size, default: 256")
    parser.add_argument("--max_t", type=int, default=10000, help="Max step in an episode, default: 10000")

    args = parser.parse_args()
    return args


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = SockShopSimulatorHyAtSSMSttAdvSAC()

    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    action1_size = env.action_space[0]
    action2_size = env.action_space[1]
    action_size = action1_size.n * action2_size.n

    agent = SAC(state_size=env.observation_space.shape[0],
                action_size=action_size,
                device=device, epsilon=0.0001)

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)

    # collect_random(env=env, dataset=buffer, num_samples=3000)

    for i in range(1, config.episodes + 1):
        state = env.reset()
        episode_steps = 0
        rewards = 0
        # 使用tqdm来显示进度条
        for step in tqdm(range(config.max_t - 1), desc=f"Episode {i}"):
            # while True:
            action = agent.get_action(state)

            steps += 1
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            if len(buffer) > config.batch_size:
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps,
                                                                                                       buffer.sample(),
                                                                                                       gamma=0.99)

            state = next_state.copy()
            rewards += reward
            episode_steps += 1
            if done:
                break

        average10.append(rewards)
        total_steps += episode_steps
        print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps, ))

        # print({"Reward": rewards,
        #            "Average10": np.mean(average10),
        #            "Total Steps": total_steps,
        #            "Policy Loss": policy_loss,
        #            "Alpha Loss": alpha_loss,
        #            "Bellmann error 1": bellmann_error1,
        #            "Bellmann error 2": bellmann_error2,
        #            "Alpha": current_alpha,
        #            "Steps": steps,
        #            "Episode": i,
        #            "Buffer size": buffer.__len__()})

        if i % config.save_every == 0:
            save(config, save_name="AdvSAC", model=agent.actor_local, ep=i)


if __name__ == "__main__":
    config = get_config()
    train(config)
