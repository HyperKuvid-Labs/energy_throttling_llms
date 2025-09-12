import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeculativeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        super(SpeculativeModel, self).__init__()
        self.w1 = nn.Linear(input_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, hidden_size)
        self.w4 = nn.Linear(hidden_size, hidden_size)
        self.w5 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        out = self.w1(x)
        out = self.relu(out)
        out = self.w2(out)
        out = self.relu(out)
        out = self.w3(out)
        out = self.relu(out)
        out = self.w4(out)
        out = self.relu(out)
        out = self.w5(out)
        out = F.softmax(out, dim=1)
        return out
    
    def update_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item() 
    
    def set_learning_rate(self, new_alpha):
        self.learning_rate = new_alpha
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_alpha

def main():
    input_size = 5
    output_size = 3
    hidden_size = 128
    model = SpeculativeModel(input_size, hidden_size, output_size)
    print(model)
    
    # # Test the model
    # test_input = torch.randn(2, input_size)
    # output = model(test_input)
    # print(f"Input shape: {test_input.shape}")
    # print(f"Output shape: {output.shape}")
    # print(f"Output probabilities sum: {torch.sum(output, dim=1)}")
    
    # # Test dynamic learning rate change
    # print(f"Current learning rate: {model.learning_rate}")
    # model.set_learning_rate(0.01)
    # print(f"New learning rate: {model.learning_rate}")

if __name__ == "__main__":
    main()