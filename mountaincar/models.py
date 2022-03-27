import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 2
OUTPUT_SIZE = 3


class MountainCarAgent(nn.Module):
    def __init__(self, flags):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE * 2, flags.hidden_layer_size)
        self.fc2 = nn.Linear(flags.hidden_layer_size, flags.hidden_layer_size // 2)
        self.output = nn.Linear(flags.hidden_layer_size // 2, OUTPUT_SIZE)

    def forward(self, inputs, goals):
        x = torch.cat([inputs.float(), goals.float()], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)
