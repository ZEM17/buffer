import numpy as np
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        
        # 初始化存储容器
        self.states = np.zeros((buffer_size, 7, 8), dtype=np.float32)  #状态 (7,8）
        self.actions_1 = np.zeros(buffer_size, dtype=np.int32)  # 动作1（6种情况）
        self.actions_2 = np.zeros(buffer_size, dtype=np.int32)  # 动作2（5种情况）
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, 7, 8), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        
        self.current_idx = 0
        self.current_size = 0

    def add(self, state, action, reward, next_state, done):
        # 分解动作
        action_1, action_2 = action
        
        # 存储经验
        self.states[self.current_idx] = state
        self.actions_1[self.current_idx] = action_1
        self.actions_2[self.current_idx] = action_2
        self.rewards[self.current_idx] = reward
        self.next_states[self.current_idx] = next_state
        self.dones[self.current_idx] = done
        
        # 更新指针和缓冲区大小
        self.current_idx = (self.current_idx + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, batch_size):
        # 随机抽样索引
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        
        # 组合动作
        actions = np.stack([
            self.actions_1[indices],
            self.actions_2[indices]
        ], axis=1)
        
        return (
            self.states[indices],
            actions,
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

# # 使用示例
# if __name__ == "__main__":
#     # 初始化缓冲区
#     buffer = ReplayBuffer(1000)
    
#     # 添加示例数据
#     for _ in range(5):
#         state = np.random.rand(7, 8)           # 随机状态
#         action = (random.randint(0,5), random.randint(0,4))  # 随机动作
#         reward = random.random()               # 随机奖励
#         next_state = np.random.rand(7, 8)      # 随机下一状态
#         done = random.choice([True, False])    # 随机结束标志
        
#         buffer.add(state, action, reward, next_state, done)
    
#     # 采样小批量数据
#     states, actions, rewards, next_states, dones = buffer.sample(2)
    
#     print(actions)