import numpy as np
import torch
import random
from algorithm.sac import SAC
from replay_buffer import ReplayBuffer
from env import ABREnv

S_DIM = [7, 8]
A_DIM = (6, 5)
RANDOM_SEED = 17
BUFFER_SIZE = int(1e6)
LR = 3e-4
TOTAL_STEPS = int(1e8)
START_STEPS = int(1e3)
UPDATE_FREQUENCY = 4
TARGET_UPDATE_FREQUENCY = 5000
TAU = 1.0  # target smoothing coefficient
BATCH_SIZE = 64
LOG_INTERVAL = 500

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
buffer = ReplayBuffer(BUFFER_SIZE)
agent = SAC(A_DIM, LR)
env = ABREnv(RANDOM_SEED)

obs = env.reset()
episode_reward = 0
episode_length = 0
for step in range(TOTAL_STEPS):
    if step <= START_STEPS:
        action = np.random.randint(30)  # 离散动作空间采样
    else:
        action, _, _ = agent.actor.get_action(obs)
    action1, action2 = env.hibrid_action(action)
    next_obs, reward, done, _ = env.step(action1, action2)
    episode_reward += reward
    episode_length += 1
    buffer.add(obs, action, reward, next_obs, done)
    obs = next_obs

    if done:
        obs = env.reset()
        episode_reward = 0
        episode_length = 0

    if step > START_STEPS:
        if step % UPDATE_FREQUENCY == 0:
            batch = buffer.sample(BATCH_SIZE)
            agent.train(batch)
    if step % UPDATE_FREQUENCY == 0:
        for param, target_param in zip(agent.qf1.parameters(), agent.qf1_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(agent.qf2.parameters(), agent.qf2_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    
    if step % LOG_INTERVAL == 0:
        print(f"Step: {step} | "
              f"Avg Reward: {episode_reward/episode_length if episode_length else 0:.6f}")