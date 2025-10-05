import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

# so the input here would would be a diff thing, profiled metrics + action taken by the target actor network

class QNetwork(nn.Module):
    def __init__(self, state_dims, action_dims=3, lr=1e-4):
        super(QNetwork, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.input_dims = state_dims + action_dims  
        self.lr = lr
        
        self.fc1 = nn.Linear(self.input_dims, int(self.input_dims * 1.5))
        self.fc2 = nn.Linear(int(self.input_dims * 1.5), int(self.input_dims * 2))
        self.fc3 = nn.Linear(int(self.input_dims * 2), int(self.input_dims * 1.5))
        self.fc4 = nn.Linear(int(self.input_dims * 1.5), 1)  
        
        self.gelu = nn.GELU()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        q_value = self.fc4(x) 
        
        return q_value
    
    def update_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
