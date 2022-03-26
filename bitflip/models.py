import torch
import torch.nn as nn
import torch.nn.functional as F


class BitflipAgent(nn.Module):
    def __init__(self, hidden_size, feature_size, bit_length):
        super().__init__()
        self.fc1 = nn.Linear(feature_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, bit_length)

    def forward(self, inputs, goals):
        input = torch.cat((inputs, goals), dim=1).float()
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Encoder(nn.Module):
    def __init__(self, hidden_size, bit_length):
        super().__init__()
        self.fc1 = nn.Linear(bit_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        return self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, hidden_size, bit_length):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, bit_length)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        return self.fc2(x)

class Dynamics(nn.Module):
    def __init__(self, hidden_size, bit_length):
        super().__init__()
        self.embeddings = nn.Embedding(bit_length, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, actions, hidden_state):
        inputs = F.relu(self.embeddings(actions))
        outputs, _ = self.gru(inputs, hidden_state[None])
        return outputs
