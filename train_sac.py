import numpy as np
import pandas as pd
import os
import torch
import random
from algorithm.sac2 import SAC
from test_sac import test
from replay_buffer import ReplayBuffer
from env import ABREnv

S_DIM = [7, 8]
A_DIM = (6, 5)
RANDOM_SEED = 1
BUFFER_SIZE = int(1e6)
LR = 1e-5
TOTAL_STEPS = 500000
START_STEPS = int(1e3)
UPDATE_FREQUENCY = 4
TARGET_UPDATE_FREQUENCY = 1000
TAU = 0.3  # target smoothing coefficient
BATCH_SIZE = 64
SAVE_INTERVAL = 1000
SAVE_PATH = "./sac_model/"

if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
buffer = ReplayBuffer(BUFFER_SIZE)
# agent = SAC(A_DIM, LR)
agent = SAC(A_DIM, S_DIM, LR)
env = ABREnv(RANDOM_SEED)

obs = env.reset()
episode_reward = 0
rewards = []
episode_length = 0
for step in range(TOTAL_STEPS):
    if step <= START_STEPS:
        action = np.random.randint(30)  # 离散动作空间采样
    else:
        action = agent.actor.get_action(obs)
    action1, action2 = env.hibrid_action(action)
    next_obs, reward, done, _ = env.step(action1, action2)
    episode_reward += reward
    episode_length += 1
    buffer.add(obs, action, reward, next_obs, done)
    obs = next_obs

    if done:
        obs = env.reset()
        rewards.append(episode_reward)
        # rewards.append(episode_reward/episode_length if episode_length else 0)
        episode_reward = 0
        episode_length = 0

    if step > START_STEPS:
        if step % UPDATE_FREQUENCY == 0:
            batch = buffer.sample(BATCH_SIZE)
            agent.train(batch)
    
    if (step > 10000) and (step % SAVE_INTERVAL == 0):
        agent.save(SAVE_PATH+"nn_model_"+str(step)+".pth")
        test_reward = test(step)
        print(f"Step: {step} | "
              f"qoe: {test_reward}")

        # data = {
        #     "Step": [step],
        #     "Avg Reward": [episode_reward/episode_length if episode_length else 0]
        # }
        # df = pd.DataFrame(data)

        # # 将数据追加到 CSV 文件
        # df.to_csv('output.csv', mode='a', header=not pd.io.common.file_exists('output.csv'), index=False)