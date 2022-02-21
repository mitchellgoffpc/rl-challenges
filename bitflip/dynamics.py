import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


def flip_bits(state, action):
    next_state = state.clone()
    next_state[action] = ~next_state[action]
    return next_state

def binary_encode(bit_length, state):
    return torch.tensor([state >> i & 1 for i in range(bit_length)], dtype=torch.bool)


class Encoder(nn.Module):
    def __init__(self, hidden_size, bit_length):
        super().__init__()
        self.fc1 = nn.Linear(bit_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, hidden_size, bit_length):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, bit_length)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Dynamics(nn.Module):
    def __init__(self, hidden_size, bit_length):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size + bit_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.samples = []

    def push(self, *args):
        if len(self) < self.max_size:
            self.samples.append(args)
        else:
            self.samples[random.randint(0, self.max_size - 1)] = args

    def sample(self, batch_size):
        sample = np.random.choice(len(self), size=batch_size, replace=False)
        return [torch.stack([self.samples[i][j] for i in sample]) for j in range(len(self.samples[0]))]

    def __len__(self):
        return len(self.samples)


# Training loop
bit_length = 32
hidden_size = 64
batch_size = 64
report_every = 1024
train_every = 16
steps = report_every / train_every

encoder = Encoder(hidden_size, bit_length)
decoder = Decoder(hidden_size, bit_length)
dynamics = Dynamics(hidden_size, bit_length)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(dynamics.parameters()), lr=0.0003)

replay_buffer = ReplayBuffer(2048)
old_state = binary_encode(bit_length, torch.randint(0, 2 ** bit_length, ()))

dynamics_error = 0
decoder_error = 0
decoder_old_accuracy = 0
decoder_new_accuracy = 0

for step in range(100000):
    action = torch.randint(0, bit_length, ())
    new_state = flip_bits(old_state, action)
    replay_buffer.push(old_state.float(), F.one_hot(action, bit_length).float(), new_state.float())
    old_state = new_state

    if len(replay_buffer) > batch_size and (step - 1) % train_every == 0:
        old_states, actions, new_states = replay_buffer.sample(batch_size)
        old_features = encoder(old_states)
        new_features = encoder(new_states)
        new_feature_preds = dynamics(old_features, actions)
        old_state_preds = decoder(old_features)
        new_state_preds = decoder(new_features)

        dynamics_loss = F.mse_loss(new_feature_preds, new_features)
        decoder_loss = F.mse_loss(old_state_preds, old_states)
        loss = dynamics_loss + decoder_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        dynamics_error += dynamics_loss.item() / steps
        decoder_error += decoder_loss.item() / steps
        decoder_old_accuracy += (old_state_preds.detach().round() == old_states).float().mean() / steps
        decoder_new_accuracy += (new_state_preds.detach().round() == new_states).float().mean() / steps

    if (step - 1) % report_every == 0:
        print(f"Dynamics error: {dynamics_error:.4f} | Decoder error: {decoder_error:.4f} | "
              f"Decoder accuracy: {decoder_old_accuracy*100:.4f}% | Dynamics + Decoder accuracy: {decoder_new_accuracy*100:.4f}%")

        dynamics_error = 0
        decoder_error = 0
        decoder_old_accuracy = 0
        decoder_new_accuracy = 0
