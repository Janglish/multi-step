import zlib
import torch
import pickle
import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    
    def __init__(self, capacity, n_step, gamma, Transition, device):
        self.Transition = Transition
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []
        self.position = 0
        self.n_step_buffer = deque(maxlen=n_step)
        self.device = device
        
    def push(self, args):
        
        self.n_step_buffer.append(self.Transition(*args))
        
        if len(self.n_step_buffer) < self.n_step:
            return
        
        if self.n_step_buffer[-1].next_state is None:
            while len(self.n_step_buffer) > 0:
                
                if len(self.buffer) < self.capacity:
                    self.buffer.append(None)
                    
                R = 0
                for i, exp in enumerate(self.n_step_buffer):
                    R += (self.gamma ** i) * exp.reward
                torch.tensor([R], device=self.device).unsqueeze(0)
                expo = torch.tensor([len(self.n_step_buffer)], device=self.device)
                experience = self.Transition(self.n_step_buffer[0].state, self.n_step_buffer[0].action, None, R, expo)
                experience = zlib.compress(pickle.dumps(experience))
                self.buffer[self.position] = experience
                self.position = (self.position + 1) % self.capacity
                self.n_step_buffer.popleft()
        else:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            R = 0
            for i, exp in enumerate(self.n_step_buffer):
                R += (self.gamma ** i) * exp.reward
            torch.tensor([R], device=self.device).unsqueeze(0)
            expo = torch.tensor([len(self.n_step_buffer)], device=self.device)
            experience = self.Transition(self.n_step_buffer[0].state, self.n_step_buffer[0].action, 
                                         self.n_step_buffer[-1].next_state, R, expo)
            experience = zlib.compress(pickle.dumps(experience))
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(np.arange(len(self.buffer)), replace=False, size=batch_size)
        experiences = tuple(pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices)
        return experiences
    
    def __len__(self):
        return len(self.buffer)