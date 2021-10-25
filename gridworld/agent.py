import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_SIZE = 5


class GridworldAgent(nn.Module):
    def __init__(self, flags):
        super().__init__()
        self.fc1 = nn.Linear(flags.grid_size * flags.grid_size * 2, flags.hidden_layer_size)
        self.fc2 = nn.Linear(flags.hidden_layer_size, flags.hidden_layer_size // 2)
        self.output = nn.Linear(flags.hidden_layer_size // 2, OUTPUT_SIZE)

    def forward(self, inputs, goals):
        x = torch.stack([inputs.float(), goals.float()], dim=-1)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.output(x), dim=-1)
