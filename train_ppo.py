import numpy as np
import pandas as pd
import os
import torch
import random
from algorithm.ppo2_baseline import Network
from test_sac import test
from replay_buffer import ReplayBuffer
from env import ABREnv

S_DIM = [7, 8]
A_DIM = 6
RANDOM_SEED = 1
LR = 1e-5
SAVE_INTERVAL = 1000
SAVE_PATH = "./sac_model/"
TRAIN_SEQ_LEN = 128 #batch size
TRAIN_EPOCH = 50000

if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# agent = SAC(A_DIM, LR)
agent = Network(S_DIM, 6, LR)
env = ABREnv(RANDOM_SEED)
rewards = []
for epoch in range(TRAIN_EPOCH):
    obs = env.reset()
    s_batch, a1_batch, a2_batch, p1_batch, p2_batch, r_batch = [], [], [], [], [], []
    for step in range(TRAIN_SEQ_LEN):
        s_batch.append(obs)

        action1_prob, action2_prob = agent.predict(
            np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

        # gumbel noise
        noise = np.random.gumbel(size=len(action1_prob))
        bit_rate = np.argmax(np.log(action1_prob) + noise)
        max_buffer_opt = np.random.choice(len(action2_prob), size=1, p=action2_prob)[0]

        obs, rew, done, info = env.step(bit_rate, max_buffer_opt)

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1
        a1_batch.append(action_vec)

        action_vec = np.zeros(5)
        action_vec[max_buffer_opt] = 1
        a2_batch.append(action_vec)

        r_batch.append(rew)
        p1_batch.append(action1_prob)
        p2_batch.append(action2_prob)
        if done:
            break
    v_batch = agent.compute_v(s_batch, r_batch, done)
    gae_batch = agent.compute_gae(r_batch, s_batch)
    # batch = [s_batch, a1_batch, a2_batch, p1_batch, p2_batch, v_batch, gae_batch]
    s_batch = np.stack(s_batch, axis=0)
    a1_batch = np.vstack(a1_batch)
    a2_batch = np.vstack(a2_batch)
    p1_batch = np.vstack(p1_batch)
    p2_batch = np.vstack(p2_batch)
    v_batch = np.vstack(v_batch)
    adv_batch = np.vstack(gae_batch)
    agent.train(s_batch, a1_batch, a2_batch, p1_batch, p2_batch, v_batch, adv_batch, epoch)

    rewards.append(np.mean(r_batch))
    if epoch % 100 == 0:  
        print("epoch:",epoch,"avg_reward:",np.mean(rewards[-10:]))