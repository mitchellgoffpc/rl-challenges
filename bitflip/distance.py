import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper functions

def flip_bits(state, action):
    next_state = state.clone()
    next_state[action] = not next_state[action]
    return next_state

def binary_encode(bit_length, i):
    return torch.tensor([i >> j & 1 for j in range(bit_length)], dtype=torch.bool)

def build_example(bit_length):
    return binary_encode(bit_length, torch.randint(0, 2 ** bit_length, ()))


class DistanceModel(nn.Module):
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
        self.bit_length = bit_length
        self.fc1 = nn.Linear(hidden_size + bit_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, states, actions):
        x = torch.cat([states, F.one_hot(actions, self.bit_length)], dim=1).float()
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


# Training

bit_length = 12
agent_hidden_size = 128
dynamics_hidden_size = 64
agent_bs = 512
dynamics_bs = 64
report_every = 1000
num_episodes = 20000

agent = DistanceModel(agent_hidden_size, dynamics_hidden_size, bit_length)
agent_optimizer = torch.optim.Adam(agent.parameters(), lr=0.0003)

encoder = Encoder(dynamics_hidden_size, bit_length)
decoder = Decoder(dynamics_hidden_size, bit_length)
dynamics = Dynamics(dynamics_hidden_size, bit_length)
dynamics_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(dynamics.parameters()), lr=0.0003)

replay_buffer = ReplayBuffer(16000)
dream_replay_buffer = ReplayBuffer(1024) # Keep the features fresh
real_wins, dream_wins, episode_lengths = 0, 0, []
decoder_errors, dynamics_errors, dynamics_accuracies = [], [], []
last_report = time.time()

for episode_counter in range(1, num_episodes + 1):
    state, goal = build_example(bit_length), build_example(bit_length)
    epsilon = max(0.05, 1. - 1.5 * float(episode_counter) / num_episodes)
    with torch.no_grad():
        goal_features = encoder(goal[None])

    # Play an episode
    for step in range(bit_length * 2):
        with torch.no_grad():
            distances = agent(encoder(state[None]), goal_features)

        # Epsilon-greedy exploration strategy
        if torch.rand(()) > epsilon:
              action = torch.argmin(distances)
        else: action = torch.randint(0, bit_length, ())

        next_state = flip_bits(state, action)
        replay_buffer.push(state, goal, action, next_state, torch.tensor(False))
        replay_buffer.push(state, next_state, action, next_state, torch.tensor(True))
        state = next_state
        if torch.all(torch.eq(next_state, goal)):
            real_wins += 1
            break

    # Play a dream episode
    goal = state = build_example(bit_length)
    with torch.no_grad():
        features = goal_features = encoder(state[None])
        for _ in range(4):
            action =  torch.randint(0, bit_length, ())
            goal = flip_bits(goal, action)
            # goal_features = dynamics(goal_features, action[None])
        goal_features = encoder(goal[None])

    for step in range(bit_length * 2):
        with torch.no_grad():
            distances = agent(features, goal_features)

        # Epsilon-greedy exploration strategy
        if torch.rand(()) > epsilon:
              action = torch.argmin(distances)
        else: action = torch.randint(0, bit_length, ())

        with torch.no_grad():
            next_state = flip_bits(state, action)
            next_features = encoder(next_state[None])
            # next_features = dynamics(features, action[None])
        dream_replay_buffer.push(features, goal_features, action, next_features, torch.tensor(False))
        dream_replay_buffer.push(features, next_features, action, next_features, torch.tensor(True))
        state = next_state
        features = next_features
        # if distances.min() < 1:
        if torch.all(torch.eq(next_state, goal)):
            dream_wins += 1
            break

    # Update the agent
    episode_lengths.append(step + 1.)

    if len(replay_buffer) > agent_bs * 4:
        states, goals, actions, next_states, finished = replay_buffer.sample(agent_bs)
        # dream_features, dream_goal_features, dream_actions, dream_next_features, dream_finished = dream_replay_buffer.sample(agent_bs)

        # Train the agent
        with torch.no_grad():
            features, next_features, goal_features = encoder(states), encoder(next_states), encoder(goals)
            best_future_distances = torch.clip(agent(next_features, goal_features).min(dim=1).values * ~finished, 0, bit_length)
        distances = agent(features, goal_features)[torch.arange(len(actions)), actions]
        loss = F.smooth_l1_loss(distances, best_future_distances + 1)
        loss.backward()

        # with torch.no_grad():
        #     best_future_distances = torch.clip(agent(dream_next_features, dream_goal_features).min(dim=1).values * ~dream_finished, 0, bit_length)
        # distances = agent(dream_features, dream_goal_features)[torch.arange(len(actions)), actions]
        # loss = F.smooth_l1_loss(distances, best_future_distances + 1)
        # loss.backward()

        agent_optimizer.step()
        agent_optimizer.zero_grad()

        # Train the dynamics model
        states, next_states, actions = states[:dynamics_bs], next_states[:dynamics_bs], actions[:dynamics_bs]
        features = encoder(states)
        next_features = encoder(next_states)
        next_feature_preds = dynamics(features, actions)
        state_preds = decoder(features)
        next_state_preds = decoder(next_features)
        dynamics_loss = F.mse_loss(next_feature_preds, next_features)
        decoder_loss = F.mse_loss(state_preds, states.float())
        loss = dynamics_loss + decoder_loss
        loss.backward()
        dynamics_optimizer.step()
        dynamics_optimizer.zero_grad()

        decoder_errors.append(decoder_loss.item())
        dynamics_errors.append(dynamics_loss.item())
        dynamics_accuracies.append((next_state_preds.detach().round() == next_states).float().mean())

    # Report stats every so often
    if episode_counter % report_every == 0:
        print(f"Episode {episode_counter:<5} | "
              f"Epsilon: {epsilon:<4.2f} | "
              f"Real Wins: {real_wins:>4} / {report_every} | "
              f"Dream Wins: {dream_wins:>4} / {report_every} | "
              f"Avg Episode Length: {np.mean(episode_lengths):.2f} | "
              f"Decoder error: {np.mean(decoder_errors):<6.4f} | "
              f"Dynamics error: {np.mean(dynamics_errors):<6.4f} | "
              f"Dynamics accuracy: {np.mean(dynamics_accuracies)*100:>6.2f}% | "
              f"Time Taken: {time.time() - last_report:.2f}s")
        real_wins, dream_wins, episode_lengths = 0, 0, []
        decoder_errors, dynamics_errors, dynamics_accuracies = [], [], []
        last_report = time.time()
