import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
    
    def __init__(self, h, w, n_actions, gamma, initial_eps, final_eps, decay_eps, seed,
                 batch_size, replay_buffer, device):
        super(Agent, self).__init__()
        self.steps_done = 0
        self.n_actions = n_actions
        self.gamma = gamma
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.decay_eps = decay_eps
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.device = device
        
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convh1 = conv2d_size_out(h, 8, 4)
        convh2 = conv2d_size_out(convh1, 4, 2)
        convh3 = conv2d_size_out(convh2, 3, 1)
        convw1 = conv2d_size_out(w, 8, 4)
        convw2 = conv2d_size_out(convw1, 4, 2)
        convw3 = conv2d_size_out(convw2, 3, 1)
        linear_input_size = convh3 * convw3 * 64
        self.linear1 = nn.Linear(linear_input_size, 512)
        nn.init.kaiming_normal_(self.linear1.weight)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(512, self.n_actions)
        nn.init.kaiming_normal_(self.linear2.weight)
        
    def forward(self, x):
        x = x.to(self.device)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.linear1(x))
        return self.linear2(x.view(x.size(0), -1))
         
    def policy(self, state):
        sample = random.random()
        eps_threshold = max(self.initial_eps - 0.9 * self.steps_done / 1000000, self.final_eps)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self(state).max(1)[1].view(1, 1)
            
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        
    def update_network(self, target_net, optimizer):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        experiences = self.replay_buffer.sample(self.batch_size)
        batch = self.replay_buffer.Transition(*zip(*experiences))
        
        non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), 
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.tensor([], device=self.device)
        batch_next_state = [s for s in batch.next_state if s is not None]
        if len(batch_next_state) != 0:
            non_final_next_states = torch.cat(batch_next_state)
        #non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).type(torch.int64)
        reward_batch = torch.cat(batch.reward)
        expo_batch = torch.cat(batch.expo)
        
        #action_batch = action_batch.type(torch.int64)
        
        state_action_values = self(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if len(non_final_next_states) != 0:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + (self.gamma ** expo_batch) * next_state_values
        
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss.backward()
        #for param in self.parameters():
        #    param.grad.data.clamp_(-1, 1)
        optimizer.step()