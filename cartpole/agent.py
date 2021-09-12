import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 4
OUTPUT_SIZE = 2


class CartPoleAgent(nn.Module):
    def __init__(self, flags):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, flags.hidden_layer_size)
        self.fc2 = nn.Linear(flags.hidden_layer_size, flags.hidden_layer_size // 2)
        self.output = nn.Linear(flags.hidden_layer_size // 2, OUTPUT_SIZE)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.float()))
        x = F.relu(self.fc2(x))
        return F.softmax(self.output(x), dim=-1)
