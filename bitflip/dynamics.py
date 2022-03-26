import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Encoder, Decoder, Dynamics
from tqdm import trange


def flip_bits(state, action):
    next_state = state.clone()
    next_state[action] = ~next_state[action]
    return next_state

def binary_encode(bit_length, state):
    return torch.tensor([state >> i & 1 for i in range(bit_length)], dtype=torch.bool)


class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.samples = []

    def push(self, *args):
        if len(self) < self.max_size:
            self.samples.append(args)
        else:
            self.samples[random.randint(0, self.max_size - 1)] = args

    def push_episode(self, episode):
        if len(episode) > 0:
            self.push(*[torch.stack([step[i] for step in episode]) for i in range(len(episode[0]))])

    def sample(self, batch_size):
        sample = np.random.choice(len(self), size=batch_size, replace=False)
        return [torch.stack([self.samples[i][j] for i in sample]) for j in range(len(self.samples[0]))]

    def __len__(self):
        return len(self.samples)


# Training loop

bit_length = 32
hidden_size = 64
batch_size = 64
report_every = 25

encoder = Encoder(hidden_size, bit_length)
decoder = Decoder(hidden_size, bit_length)
dynamics = Dynamics(hidden_size, bit_length)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(dynamics.parameters()), lr=0.0003)

replay_buffer = ReplayBuffer(1024)
decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []

for episode_counter in range(1000):
    episode = []
    old_state = binary_encode(bit_length, torch.randint(0, 2 ** bit_length, ()))
    for step in range(8):
        action = torch.randint(0, bit_length, ())
        new_state = flip_bits(old_state, action)
        episode.append((old_state, action, new_state))
        old_state = new_state
    replay_buffer.push_episode(episode)

    if len(replay_buffer) > batch_size:
        for _ in range(8):
            old_states, actions, new_states = replay_buffer.sample(batch_size)
            old_features = encoder(old_states)
            new_features = encoder(new_states)
            new_feature_preds = dynamics(actions, old_features[:,0])
            old_state_preds = decoder(old_features)
            new_state_preds = decoder(new_features)

            dynamics_loss = F.mse_loss(new_feature_preds, new_features)
            decoder_loss = F.mse_loss(old_state_preds, old_states.float())
            loss = dynamics_loss + decoder_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            dynamics_errors.append(dynamics_loss.item())
            decoder_errors.append(decoder_loss.item())
            decoder_old_accuracies.append((old_state_preds.detach().round() == old_states).float().mean())
            decoder_new_accuracies.append((new_state_preds.detach().round() == new_states).float().mean())

    if episode_counter % report_every == 0:
        print(f"Episode {episode_counter:<4} | "
              f"Dynamics error: {np.mean(dynamics_errors):.4f} | "
              f"Decoder error: {np.mean(decoder_errors):.4f} | "
              f"Decoder accuracy: {np.mean(decoder_old_accuracies)*100:.3f}% | "
              f"Dynamics + Decoder accuracy: {np.mean(decoder_new_accuracies)*100:.3f}%")

        decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []
