import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

class Observer():
    
    def __init__(self, env, device, seed):
        self.env = env
        self.device = device
        
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def step(self, action):
        action = action.item()
        next_state, reward, done, info = self.env.step(action)
        reward = torch.tensor([reward], device=self.device)
        next_state = np.ascontiguousarray(next_state, dtype=np.float32) / 255
        next_state = torch.tensor(next_state, device=self.device)
        next_state = next_state.unsqueeze(0)
        next_state = next_state.unsqueeze(0).to(self.device)
        return next_state, reward, done, info
        
    def reset(self):
        state = self.env.reset()
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        state = torch.tensor(state, device=self.device)
        state = state.unsqueeze(0)
        return state.unsqueeze(0).to(self.device)
    
    def close(self):
        return self.env.close()
        
    def render(self, mode="rgb_array"):
        return self.env.render(mode)