import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

# deciding on the structure of the target actor network, 
# input_dims -> input_dim*1.5 -> input_dims*2 -> input_dims*1.5 -> 3

class TargetActorNetwork(nn.Module):
    def __init__(self, input_dims, output_dims=3, lr=1e-5):
        super(TargetActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.lr = lr

        self.fc1 = nn.Linear(self.input_dims, int(self.input_dims * 1.5))
        self.fc2 = nn.Linear(int(self.input_dims * 1.5), int(self.input_dims * 2))
        self.fc3 = nn.Linear(int(self.input_dims * 2), int(self.input_dims * 1.5))
        self.fc4 = nn.Linear(int(self.input_dims * 1.5), self.output_dims)

        self.gelu = nn.GELU()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, metrics):
        x = self.gelu(self.fc1(metrics))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        x = self.fc4(x) 

        spec_num_steps = torch.sigmoid(x[:, 0]) * 31 + 1 # range: [1, 32]
        spec_eagle_topk = torch.sigmoid(x[:, 1]) * 9 + 1 # range: [1, 10] 
        spec_num_draft_tokens = torch.sigmoid(x[:, 2]) * 63 + 1 # range: [1, 64]

        return torch.stack([
            torch.round(spec_num_steps),
            torch.round(spec_eagle_topk), 
            torch.round(spec_num_draft_tokens)
        ], dim=1)
    
    def update_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# the three outputs over here are: speculative_num_steps, speculative_eagle_topk, speculative_num_draft_tokens
# the inputs are current hardware metrics, for cpu as well as gpu
# hardware_state = [
#     cpu_temp_current,           
#     gpu_temp_current,             
#     gpu_util_current,           
#     vram_available,             
#     ram_available,              
#     storage_available,          
#     fan_speed_current,          
#     battery_remaining,         
# ]
