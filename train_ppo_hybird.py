import numpy as np
import pandas as pd
import os
import torch
import random
from algorithm.ppo2_hybrid import Network
from test_ppo_hybrid import test
from env_continuous import ABREnv
import wandb
import os

os.environ["WANDB_API_KEY"] = "0ceea818e30b944d80ac9b37406a14ba021b7f0f"
os.environ["WANDB_MODE"] = "online"

S_DIM = [7, 8]
A_DIM = 6
RANDOM_SEED = 17
LR = 1e-4
SAVE_INTERVAL = 500
SAVE_PATH = "./ppo_hybrid_model/"
TRAIN_SEQ_LEN = 128 #batch size
TRAIN_EPOCH = 50000
FEATURE_NUM = 64

if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

run = wandb.init(
    # Set the project where this run will be logged
    project="ppo_hybrid",
    # Track hyperparameters and run metadata
    config={
         "learning rate": LR,
         "train epoch": TRAIN_EPOCH,
         "seed": RANDOM_SEED,
         "batch size": TRAIN_SEQ_LEN,
         "feature num": FEATURE_NUM
    },
)

# agent = SAC(A_DIM, LR)
agent = Network(S_DIM, 6, LR, FEATURE_NUM)
env = ABREnv(RANDOM_SEED)
rewards = []
for epoch in range(TRAIN_EPOCH):
    obs = env.reset()
    s_batch, a1_batch, a2_batch, p1_batch, p2_log_batch, r_batch = [], [], [], [], [], []
    for step in range(TRAIN_SEQ_LEN):
        s_batch.append(obs)

        # action1_prob, action2_prob = agent.predict(
        #     np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
        action1_prob, action2, action2_log = agent.predict(
            np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
        # gumbel noise
        noise = np.random.gumbel(size=len(action1_prob))
        bit_rate = np.argmax(np.log(action1_prob) + noise)

        max_buffer_opt = action2[0]

        obs, rew, done, info = env.step(bit_rate, max_buffer_opt)

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1
        a1_batch.append(action_vec)
        a2_batch.append(max_buffer_opt)

        r_batch.append(rew)
        p1_batch.append(action1_prob)
        p2_log_batch.append(action2_log)
        if done:
            break
    v_batch = agent.compute_v(s_batch, r_batch, done)
    gae_batch = agent.compute_gae(r_batch, s_batch)
    # batch = [s_batch, a1_batch, a2_batch, p1_batch, p2_batch, v_batch, gae_batch]
    s_batch = np.stack(s_batch, axis=0)
    a1_batch = np.vstack(a1_batch)
    a2_batch = np.vstack(a2_batch)
    p1_batch = np.vstack(p1_batch)
    p2_log_batch = np.vstack(p2_log_batch)
    v_batch = np.vstack(v_batch)
    adv_batch = np.vstack(gae_batch)
    loss_a1_value, loss_a2_value, loss_critic_value, loss_total_value, entropy_weight= agent.train(s_batch, a1_batch, a2_batch, p1_batch, p2_log_batch, v_batch, adv_batch, epoch)

    rewards.append(np.mean(r_batch))
    
    if epoch % SAVE_INTERVAL == 0:
        agent.save_model(SAVE_PATH+"nn_model_"+str(epoch)+".pth")
        qoe, data_size, buffer = test(epoch)
        print("epoch: %d\n" \
              "qoe: %.3f, reward: %.3f\n" \
              "buffer: %.3f, data_size: %.3f\n" \
              "loss: %.3f, entropy_weight: %.3f\n\n" \
              % (epoch, qoe, (qoe/data_size), buffer, data_size, loss_total_value, entropy_weight))
        log_info = {
            "qoe": qoe, 
            "reward": qoe/data_size,
            "buffer": buffer,
            "data_size": data_size,
            "loss": loss_total_value, 
            "loss_a1": loss_a1_value,
            "loss_a2": loss_a2_value,
            "loss_critic_value": loss_critic_value,
            "entropy_weight": entropy_weight
            }
        wandb.log(log_info)