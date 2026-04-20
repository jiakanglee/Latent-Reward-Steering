import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)
        
    def forward(self, x):
        return self.linear(x)