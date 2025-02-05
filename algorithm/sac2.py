import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

FEATURE_NUM = 256


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
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
    
    def get_action(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = np.reshape(state,(1, 7, 8))
                state = torch.FloatTensor(state).to(next(self.parameters()).device)
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
        return action.cpu().item()

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
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

class SAC:
    def __init__(self, action_dim, state_dim, lr=1e-3):
        a1_dim, a2_dim = action_dim
        self.action_dim = a1_dim * a2_dim
        self.state_dim = state_dim
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Actor网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic网络
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # 自动调节温度参数
        self.target_entropy = torch.log(torch.tensor(self.action_dim)).to(self.device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp()

        # 超参数
        self.gamma = 0.99
        self.tau = 0.3

    def train(self, batch):
        state, action, reward, next_state, done = batch

        # 转换数据为tensor
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)  # 离散动作需要long类型
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # 更新Critic网络
        with torch.no_grad():
            next_logits = self.actor(next_state)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            q1_next = self.critic1_target(next_state)
            q2_next = self.critic2_target(next_state)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done) * self.gamma * (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)

        current_q1 = self.critic1(state).gather(1, action)
        current_q2 = self.critic2(state).gather(1, action)
        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic2_optimizer.step()

        # 更新Actor网络
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        q1 = self.critic1(state)
        q2 = self.critic2(state)
        min_q = torch.min(q1, q2)
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新温度参数alpha
        entropy = -(probs * log_probs).sum(dim=1)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 软更新目标网络
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        """保存模型参数到指定路径"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha.data,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, path)
        # print(f"模型保存到 {path}")

    def load(self, path):
        """从指定路径加载模型参数"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        # 加载网络参数
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        # 加载优化器状态
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        # 加载alpha相关参数
        self.log_alpha.data = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.alpha = self.log_alpha.exp()
        # print(f"从 {path} 加载模型成功")