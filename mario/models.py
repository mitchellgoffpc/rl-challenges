import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

INPUT_SIZE = (3, 224, 240)
NUM_ACTIONS = 8


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, hidden_size)

    def forward(self, x):
        *b,c,h,w = x.shape
        x = self.backbone(x.view(-1,c,h,w).float())  # No relu for now
        x = self.fc(x.flatten(start_dim=1))
        return x.view(*b,-1)

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        c,h,w = INPUT_SIZE
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, c*h*w)

    def forward(self, x):
        # x = F.relu(self.fc1(x.float()))
        # return self.fc2(x).view(x.shape[0], x.shape[1], self.width, self.height, NUM_CHANNELS)
        return x

class Dynamics(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(NUM_ACTIONS, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, actions, hidden_state):
        inputs = F.relu(self.fc(actions.float()))
        outputs, _ = self.gru(inputs, hidden_state[None])
        return outputs
