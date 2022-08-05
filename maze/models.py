import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CHANNELS = 5
NUM_ACTIONS = 4


class MazeAgent(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super().__init__()
        self.fc1 = nn.Linear(feature_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, NUM_ACTIONS)

    def forward(self, inputs, goals):
        input = torch.cat((inputs, goals), dim=1).float()
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Encoder(nn.Module):
    def __init__(self, hidden_size, width, height):
        super().__init__()
        # self.conv = nn.Conv2d(5, 16, 1)
        # self.fc1 = nn.Linear(width*height*16, hidden_size)
        self.fc1 = nn.Linear(width*height*NUM_CHANNELS, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        *b,h,w,c = x.shape
        # x = F.relu(self.conv(x.view(-1,h,w,c).permute(0,3,1,2).float()))
        x = F.relu(self.fc1(x.float().flatten(start_dim=-3)))
        return self.fc2(x).view(*b,-1)

class Decoder(nn.Module):
    def __init__(self, hidden_size, width, height):
        super().__init__()
        self.width, self.height = width, height
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, width*height*NUM_CHANNELS)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        return self.fc2(x).view(x.shape[0], x.shape[1], self.width, self.height, NUM_CHANNELS)

class Dynamics(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embeddings = nn.Embedding(NUM_ACTIONS, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, actions, hidden_state):
        inputs = F.relu(self.embeddings(actions))
        outputs, _ = self.gru(inputs, hidden_state[None])
        return outputs

class Finished(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super().__init__()
        self.fc1 = nn.Linear(feature_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs, goals):
        input = torch.cat((inputs, goals), dim=1).float()
        x = F.relu(self.fc1(input))
        return self.fc2(x)
