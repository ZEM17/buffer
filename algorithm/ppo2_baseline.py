import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np

ACTION_EPS = 1e-4
GAMMA = 0.99
EPS = 0.2  # PPO2 epsilon

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, feature_num):
        super(Actor, self).__init__()
        # Actor network
        self.s_dim = state_dim
        self.a1_dim = action_dim
        self.a2_dim = 5

        self.fc1_actor = nn.Linear(1, feature_num)
        self.fc2_actor = nn.Linear(1, feature_num)
        self.conv3_actor = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=feature_num)
        self.conv4_actor = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=feature_num)
        self.conv5_actor = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=feature_num)
        self.fc6_actor = nn.Linear(1, feature_num)
        self.fc7_actor = nn.Linear(1, feature_num)
        self.bitrate_action = nn.Linear(1664, feature_num)
        self.max_buffer_action = nn.Linear(1664, feature_num)
        self.auxiliary_action = nn.Linear(1664, feature_num)

        self.bitrate_pi_head = nn.Linear(feature_num, self.a1_dim)
        self.max_buffer_pi_head = nn.Linear(feature_num, self.a2_dim)
        self.auxiliary_pi_head = nn.Linear(feature_num, 1)

        # self.optimizer = optim.Adam(list(self.parameters()), lr=learning_rate)

    def forward(self, inputs):
        split_1 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_2 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_3 = F.relu(self.conv3_actor(inputs[:, 2:3, :])).view(inputs.shape[0], -1)
        split_4 = F.relu(self.conv4_actor(inputs[:, 3:4, :]).view(inputs.shape[0], -1))
        split_5 = F.relu(self.conv5_actor(inputs[:, 4:5, :self.a1_dim]).view(inputs.shape[0], -1))
        split_6 = F.relu(self.fc6_actor(inputs[:, 5:6, -1]))
        split_7 = F.relu(self.fc7_actor(inputs[:, 6:7, -1]))

        merge_net = torch.cat([split_1, split_2, split_3, split_4, split_5, split_6, split_7], 1)

        a1 = self.bitrate_action(merge_net)
        a1 = F.softmax(self.bitrate_pi_head(a1), dim=-1)
        a1 = torch.clamp(a1, ACTION_EPS, 1. - ACTION_EPS)

        a2 = self.max_buffer_action(merge_net)
        a2 = F.softmax(self.max_buffer_pi_head(a2), dim=-1)
        a2 = torch.clamp(a2, ACTION_EPS, 1. - ACTION_EPS)

        a3 = self.auxiliary_action(merge_net)
        a3 = self.auxiliary_pi_head(a3)
        return a1, a2, a3


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, feature_num):
        super(Critic, self).__init__()
        # Critic network
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fc1_critic = nn.Linear(1, feature_num)
        self.fc2_critic = nn.Linear(1, feature_num)
        self.conv3_critic = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=feature_num)
        self.conv4_critic = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=feature_num)
        self.conv5_critic = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=feature_num)
        self.fc6_critic = nn.Linear(1, feature_num)
        self.fc7_critic = nn.Linear(1, feature_num)
        self.merge_critic = nn.Linear(1664, feature_num)
        self.val_head = nn.Linear(feature_num, 1)

        # self.optimizer = optim.Adam(list(self.parameters()), lr=learning_rate)

    def forward(self, inputs):
        split_1 = F.relu(self.fc1_critic(inputs[:, 0:1, -1]))
        split_2 = F.relu(self.fc2_critic(inputs[:, 1:2, -1]))
        split_3 = F.relu(self.conv3_critic(inputs[:, 2:3, :])).view(inputs.shape[0], -1)
        split_4 = F.relu(self.conv4_critic(inputs[:, 3:4, :]).view(inputs.shape[0], -1))
        split_5 = F.relu(self.conv5_critic(inputs[:, 4:5, :self.a_dim]).view(inputs.shape[0], -1))
        split_6 = F.relu(self.fc6_critic(inputs[:, 5:6, -1]))
        split_7 = F.relu(self.fc7_critic(inputs[:, 6:7, -1]))

        merge_net = torch.cat([split_1, split_2, split_3, split_4, split_5, split_6, split_7], 1)

        value_net = F.relu(self.merge_critic(merge_net))
        value = self.val_head(value_net)
        return value
    
class Network():
    def __init__(self, state_dim, action_dim, learning_rate, feature_num):

        self.s_dim = state_dim
        self.action_dim = action_dim
        self._entropy_weight = np.log(action_dim)
        self.H_target = 0.1
        self.PPO_TRAINING_EPO = 5
        self.AUX_TRAINING_EPO = 6

        self.actor = Actor(state_dim, action_dim, feature_num).to(device)
        self.critic = Critic(state_dim, action_dim, feature_num).to(device)
        self.lr_rate = learning_rate
        self.optimizer = optim.Adam(list(self.actor.parameters()) + \
                                    list(self.critic.parameters()), lr=learning_rate)

    def get_network_params(self):
        return [self.actor.state_dict(), self.critic.state_dict()]
    
    def set_network_params(self, input_network_params):
        actor_net_params, critic_net_params = input_network_params
        self.actor.load_state_dict(actor_net_params)
        self.critic.load_state_dict(critic_net_params)

    def r(self, pi_new, pi_old, acts):
        return torch.sum(pi_new * acts, dim=1, keepdim=True) / \
               torch.sum(pi_old * acts, dim=1, keepdim=True)

    def train(self, s_batch, a1_batch, a2_batch, p1_batch, p2_batch, v_batch, adv_batch, epoch):
        s_batch = torch.from_numpy(s_batch).to(torch.float32).to(device)
        a1_batch = torch.from_numpy(a1_batch).to(torch.float32).to(device)
        a2_batch = torch.from_numpy(a2_batch).to(torch.float32).to(device)
        p1_batch = torch.from_numpy(p1_batch).to(torch.float32).to(device)
        p2_batch = torch.from_numpy(p2_batch).to(torch.float32).to(device)
        v_batch = torch.from_numpy(v_batch).to(torch.float32).to(device)
        adv_batch = torch.from_numpy(adv_batch).to(torch.float32).to(device)
        
        loss_a1_value = 0
        loss_a2_value = 0
        loss_critic_value = 0
        loss_total_value = 0
        # policy phase
        for _ in range(self.PPO_TRAINING_EPO):
            pi1, pi2, _ = self.actor.forward(s_batch)
            val = self.critic.forward(s_batch)

            # loss1
            adv = adv_batch
            ratio1 = self.r(pi1, p1_batch, a1_batch)
            ppo2loss1 = torch.min(ratio1 * adv, torch.clamp(ratio1, 1 - EPS, 1 + EPS) * adv)
            # Dual-PPO
            dual_loss1 = torch.where(adv < 0, torch.max(ppo2loss1, 3. * adv), ppo2loss1)
            loss1_entropy = torch.sum(-pi1 * torch.log(pi1), dim=1, keepdim=True)
            loss1 = -dual_loss1.mean() - self._entropy_weight * loss1_entropy.mean()

            # loss2
            ratio2 = self.r(pi2, p2_batch, a2_batch)
            ppo2loss2 = torch.min(ratio2 * adv, torch.clamp(ratio2, 1 - EPS, 1 + EPS) * adv)
            # Dual-PPO
            dual_loss2 = torch.where(adv < 0, torch.max(ppo2loss2, 3. * adv), ppo2loss2)
            loss2_entropy = torch.sum(-pi2 * torch.log(pi2), dim=1, keepdim=True)
            loss2 = -dual_loss2.mean() - self._entropy_weight * loss2_entropy.mean()

            # vloss
            loss3 = F.mse_loss(val, v_batch)

            loss = loss1 + loss2 + loss3

            loss_a1_value = loss1.item()
            loss_a2_value = loss2.item()
            loss_critic_value = loss3.item()
            loss_total_value = loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()





        # auxiliary phase
        s = s_batch
        v_target = self.critic.forward(s)
        pi1_old, pi2_old, _ = self.actor.forward(s)

        pi1_old, pi2_old = pi1_old.detach(), pi2_old.detach()
        v_target = v_target.detach()

        for _ in range(self.AUX_TRAINING_EPO):
            pi1, pi2, v3 = self.actor.forward(s)
            loss_aux = F.mse_loss(v_target, v3)

            kl1 = torch.sum(pi1_old * torch.log(pi1_old / pi1))
            kl2 = torch.sum(pi2_old * torch.log(pi2_old / pi2))
            loss_joint = loss_aux + kl1 + kl2
            self.optimizer.zero_grad()
            loss_joint.backward()
            self.optimizer.step()
        



        # Update entropy weight
        _H = (-(torch.log(p1_batch) * p1_batch).sum(dim=1)).mean().item()
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
        self._entropy_weight = max(self._entropy_weight, 1e-2)
        
        return loss_a1_value, loss_a2_value, loss_critic_value, loss_total_value, self._entropy_weight

    def predict(self, input):
        with torch.no_grad():
            input = torch.from_numpy(input).to(torch.float32).to(device)
            pi1 = self.actor.forward(input)[0][0]
            pi2 = self.actor.forward(input)[1][0]
            return pi1.cpu().numpy(), pi2.cpu().numpy()

    def load_model(self, nn_model):
        actor_model_params, critic_model_params = torch.load(nn_model, weights_only=True, map_location=torch.device('cpu'))
        self.actor.load_state_dict(actor_model_params)
        self.critic.load_state_dict(critic_model_params)

    def save_model(self, nn_model):
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        torch.save(model_params, nn_model)

    def compute_v(self, s_batch, r_batch, terminal):
        R_batch = np.zeros_like(r_batch)

        if terminal:
            # in this case, the terminal reward will be assigned as r_batch[-1]
            R_batch[-1] = r_batch[-1]  # terminal state
        else:
            val = self.critic.forward(s_batch)
            R_batch[-1] = val[-1]  # bootstrap from last state

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t] = r_batch[t] + GAMMA * R_batch[t + 1]

        return list(R_batch)

    def compute_gae(self, r, s, lam=0.95):
        s_batch = np.stack(s, axis=0)
        s_batch = torch.from_numpy(s_batch).to(torch.float32).to(device)
        values = self.critic.forward(s_batch)
        a = torch.tensor([[0.]]).to(device)
        values = torch.cat([values, a], dim=0)
        rewards = r

        T = len(rewards)  # 轨迹的长度
        advantages = np.zeros(T)  # 存储每一步的优势估计
        gae = 0  # 初始化 GAE 累积变量

        for t in reversed(range(T)):  # 从最后一步开始计算
            # TD 误差 δ_t
            delta = rewards[t] + GAMMA * values[t+1] - values[t]
            # GAE 的递推公式
            gae = delta + GAMMA * lam * gae
            advantages[t] = gae

        return list(advantages)
           
