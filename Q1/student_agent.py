import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# PolicyNet 從 train.py 複製而來（略做簡化）
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
        self.scale = (self.action_high - self.action_low) / 2.0
        self.bias = (self.action_high + self.action_low) / 2.0

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self(state)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        scale = self.scale.to(y_t.device)
        bias = self.bias.to(y_t.device)
        action = scale * y_t + bias
        return action


# Agent 載入訓練好的 pt 檔案並執行動作
class Agent(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make("Pendulum-v1")
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        self.policy = PolicyNet(
            state_dim=obs_space.shape[0],
            action_dim=act_space.shape[0],
            action_low=act_space.low,
            action_high=act_space.high
        ).to(self.device)

        # 載入訓練好的 actor 權重檔案
        self.policy.load_state_dict(torch.load("value.pt", map_location=self.device))
        self.policy.eval()

    def act(self, observation):
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy.sample(state)
        return action.squeeze(0).cpu().numpy()
