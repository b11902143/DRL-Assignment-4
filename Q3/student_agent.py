# student_agent.py

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor network must match the one used in training
class Actor(nn.Module):
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.l1       = nn.Linear(state_dim, hidden)
        self.l2       = nn.Linear(hidden, hidden)
        self.mean     = nn.Linear(hidden, action_dim)
        self.log_std  = nn.Linear(hidden, action_dim)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        m = self.mean(x)
        # log_std is unused at inference
        return m

class Agent(object):
    """Load the trained SAC actor from 'value.pth' and run a deterministic policy."""
    def __init__(self):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Humanoid Walk dims
        state_dim  = 67
        action_dim = 21

        # build actor and load weights
        self.actor = Actor(state_dim, action_dim).to(self.device)
        ckpt_path = "value.pth"
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found in Q3 directory")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        # if checkpoint is a dict with 'actor' key, extract it
        if isinstance(ckpt, dict) and "actor" in ckpt:
            state_dict = ckpt["actor"]
        else:
            # assume ckpt is the raw state_dict
            state_dict = ckpt
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

        # action space for conformity
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def act(self, observation):
        # convert obs to tensor
        s = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            m = self.actor(s)
            # deterministic: use mean, apply tanh to bound [-1,1]
            a = torch.tanh(m)
        return a.squeeze(0).cpu().numpy()
