import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FEATURE_NUM = 256
ALPHA = 0.2
GAMMA = 0.99
TARGET_ENTROPY_SCALE = 0.89
S_DIM = [7, 8]


class SoftQNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.a1_dim, self.a2_dim = action_dim
        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv3_actor = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=FEATURE_NUM)
        self.conv4_actor = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=FEATURE_NUM)
        self.conv5_actor = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=FEATURE_NUM)
        self.fc6_actor = nn.Linear(1, FEATURE_NUM)
        self.fc7_actor = nn.Linear(1, FEATURE_NUM)
 
        self.fc_q = nn.Sequential(
            nn.Linear(6656, FEATURE_NUM),
            nn.Linear(FEATURE_NUM, self.a1_dim * self.a2_dim)
        )

    def forward(self, x):
        split_1 = F.relu(self.fc1_actor(x[:, 0:1, -1]))
        split_2 = F.relu(self.fc2_actor(x[:, 1:2, -1]))
        split_3 = F.relu(self.conv3_actor(x[:, 2:3, :])).view(x.shape[0], -1)
        split_4 = F.relu(self.conv4_actor(x[:, 3:4, :]).view(x.shape[0], -1))
        split_5 = F.relu(self.conv5_actor(x[:, 4:5, :self.a1_dim]).view(x.shape[0], -1))
        split_6 = F.relu(self.fc6_actor(x[:, 5:6, -1]))
        split_7 = F.relu(self.fc7_actor(x[:, 6:7, -1]))

        merge_net = torch.cat([split_1, split_2, split_3, split_4, split_5, split_6, split_7], 1)

        q = self.fc_q(merge_net)

        return q


class Actor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.a1_dim, self.a2_dim = action_dim
        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv3_actor = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=FEATURE_NUM)
        self.conv4_actor = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=FEATURE_NUM)
        self.conv5_actor = nn.Conv1d(kernel_size=1, in_channels=1, out_channels=FEATURE_NUM)
        self.fc6_actor = nn.Linear(1, FEATURE_NUM)
        self.fc7_actor = nn.Linear(1, FEATURE_NUM)
 
        self.fc_logits = nn.Sequential(
            nn.Linear(6656, FEATURE_NUM),
            nn.Linear(FEATURE_NUM, self.a1_dim * self.a2_dim)
        )

    def forward(self, x):

        split_1 = F.relu(self.fc1_actor(x[:, 0:1, -1]))
        split_2 = F.relu(self.fc2_actor(x[:, 1:2, -1]))
        split_3 = F.relu(self.conv3_actor(x[:, 2:3, :])).view(x.shape[0], -1)
        split_4 = F.relu(self.conv4_actor(x[:, 3:4, :]).view(x.shape[0], -1))
        split_5 = F.relu(self.conv5_actor(x[:, 4:5, :self.a1_dim]).view(x.shape[0], -1))
        split_6 = F.relu(self.fc6_actor(x[:, 5:6, -1]))
        split_7 = F.relu(self.fc7_actor(x[:, 6:7, -1]))

        merge_net = torch.cat([split_1, split_2, split_3, split_4, split_5, split_6, split_7], 1)

        logits = self.fc_logits(merge_net)

        return logits

    def get_action(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.reshape(x,(1, S_DIM[0], S_DIM[1]))
            x = torch.from_numpy(x).to(torch.float32).to(device)
        logits = self.forward(x)
        # 创建分类分布并采样
        dist = torch.distributions.Categorical(logits=logits)
        
        action = dist.sample()

        # 计算对数概率（用于SAC的熵正则化）
        log_prob = F.log_softmax(logits, dim=1)

        action_probs = dist.probs
        
        return action, log_prob, action_probs

class SAC():
    def __init__(self, action_dim, lr):
        self.actor = Actor(action_dim).to(device)
        self.qf1 = SoftQNetwork(action_dim).to(device)
        self.qf2 = SoftQNetwork(action_dim).to(device)
        self.qf1_target = SoftQNetwork(action_dim).to(device)
        self.qf2_target = SoftQNetwork(action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=lr, eps=1e-4)
        self.alpha = ALPHA
        self.target_entropy = TARGET_ENTROPY_SCALE * torch.log(1 / torch.tensor(action_dim[0] * action_dim[1]).to(device))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=lr, eps=1e-4)

    def cac_target(self, rewards, next_states, dones):
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor.get_action(next_states)
            qf1_next_target = self.qf1_target(next_states)
            qf2_next_target = self.qf2_target(next_states)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * GAMMA * (min_qf_next_target)
            return next_q_value

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.from_numpy(states).to(torch.float32).to(device)
        next_states = torch.from_numpy(next_states).to(torch.float32).to(device)
        rewards = torch.from_numpy(rewards).to(torch.float32).to(device)
        dones = torch.from_numpy(dones).to(torch.int64).to(device)
        actions = torch.from_numpy(actions).to(torch.int64).to(device)

        next_q_value = self.cac_target(rewards, next_states, dones)

        # critic training
        qf1_values = self.qf1(states)
        qf2_values = self.qf2(states)
        qf1_a_values = qf1_values.gather(1, actions.unsqueeze(1)).view(-1)
        qf2_a_values = qf2_values.gather(1, actions.unsqueeze(1)).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        
        # actor training
        _, log_pi, action_probs = self.actor.get_action(states)
        with torch.no_grad():
            qf1_values = self.qf1(states)
            qf2_values = self.qf2(states)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'qf1_target_state_dict': self.qf1_target.state_dict(),
            'qf2_target_state_dict': self.qf2_target.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'a_optimizer_state_dict': self.a_optimizer.state_dict(),
            'log_alpha': self.log_alpha
        }, path)
        # print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.qf1.load_state_dict(checkpoint['qf1_state_dict'])
        self.qf2.load_state_dict(checkpoint['qf2_state_dict'])
        self.qf1_target.load_state_dict(checkpoint['qf1_target_state_dict'])
        self.qf2_target.load_state_dict(checkpoint['qf2_target_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha'].to(device)
        self.log_alpha.requires_grad_(True)  # 确保梯度跟踪
        self.alpha = self.log_alpha.exp().item()
        # print(f"Model loaded from {path}")