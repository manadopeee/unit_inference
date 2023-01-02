import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.fc1 = nn.Linear(34, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(classes))

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.dropout(F.relu(self.fc2(x)), 0.5)
        x = self.fc3(x)
        return x
