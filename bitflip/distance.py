import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper functions

def flip_bits(state, action):
    next_state = state.clone()
    if state.ndim == 1:
          next_state[action] = not next_state[action]
    else: next_state[torch.arange(len(next_state)), action] = ~next_state[torch.arange(len(next_state)), action]
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
        self.embeddings = nn.Embedding(bit_length, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, actions, hidden_state):
        inputs = F.relu(self.embeddings(actions))
        outputs, _ = self.gru(inputs, hidden_state[None])
        return outputs

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
        sample = random.sample(range(len(self)), batch_size)
        return [torch.stack([self.samples[i][col] for i in sample]) for col in range(len(self.samples[0]))]

    def __len__(self):
        return len(self.samples)


bit_length = 8
agent_hidden_size = 128
dynamics_hidden_size = 64
agent_bs = 512
dynamics_bs = 64
report_every = 1000
num_episodes = 20000
max_episode_length = bit_length * 2
max_dream_length = 2

agent = DistanceModel(agent_hidden_size, dynamics_hidden_size, bit_length)
agent_optimizer = torch.optim.Adam(agent.parameters(), lr=0.0003)

encoder = Encoder(dynamics_hidden_size, bit_length)
decoder = Decoder(dynamics_hidden_size, bit_length)
dynamics = Dynamics(dynamics_hidden_size, bit_length)
dynamics_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(dynamics.parameters()), lr=0.0003)

agent_replay_buffer = ReplayBuffer(16000)
dynamics_replay_buffer = ReplayBuffer(1024)
real_wins, dream_wins, dreams, episode_lengths = 0, 0, 0, []
decoder_errors, dynamics_errors, dynamics_accuracies = [], [], []
rolling_dynamics_accuracy = 0.0
last_report = time.time()


# Training loop

for episode_counter in range(1, num_episodes + 1):
    state, goal = build_example(bit_length), build_example(bit_length)
    epsilon = max(0.00, 1. - 3 * float(episode_counter) / num_episodes)
    with torch.no_grad():
        goal_features = encoder(goal[None])

    # Play an episode
    episode = []
    for step in range(max_episode_length):
        with torch.no_grad():
            distances = agent(encoder(state[None]), goal_features)

        # Epsilon-greedy exploration strategy
        if torch.rand(()) > epsilon:
              action = torch.argmin(distances)
        else: action = torch.randint(0, bit_length, ())

        next_state = flip_bits(state, action)
        agent_replay_buffer.push(state, goal, action, next_state, torch.tensor(False))
        agent_replay_buffer.push(state, next_state, action, next_state, torch.tensor(True))
        episode.append((state, action, next_state))
        state = next_state
        if torch.all(torch.eq(next_state, goal)):
            real_wins += 1
            break

    episode_lengths.append(step + 1.)
    if len(episode) > 0:
        mask = torch.zeros(max_episode_length, dtype=torch.bool)
        mask[:len(episode)] = True
        columns = [torch.stack([episode[i][j] if mask[i] else episode[0][j]
            for i in range(max_episode_length)])
            for j in range(len(episode[0]))]
        dynamics_replay_buffer.push(*columns, mask)


    # Train the agent on real episodes
    if len(agent_replay_buffer) > agent_bs:
        states, goals, actions, next_states, finished = agent_replay_buffer.sample(agent_bs)

        with torch.no_grad():
            features, next_features, goal_features = encoder(states), encoder(next_states), encoder(goals)
            best_future_distances = torch.clip(agent(next_features, goal_features).min(dim=1).values * ~finished, 0, bit_length)
        distances = agent(features, goal_features)[torch.arange(len(actions)), actions]
        loss = F.smooth_l1_loss(distances, best_future_distances + 1)
        loss.backward()

        agent_optimizer.step()
        agent_optimizer.zero_grad()

    # Train the agent on dream episodes
    if len(agent_replay_buffer) > agent_bs:
        states, _, _, _, _ = agent_replay_buffer.sample(agent_bs)
        with torch.no_grad():
            features = encoder(states)

        # Single-step
        with torch.no_grad():
            actions = torch.randint(0, bit_length, (len(states),))
            goal_features = dynamics(actions[:,None], features)[:,0]
        distances = agent(features, goal_features)
        loss = F.smooth_l1_loss(distances[torch.arange(len(actions)), actions], torch.ones(len(actions)))
        loss.backward()

        # Multi-step
        with torch.no_grad():
            goals, goal_features = states, features
            # lengths = torch.randint(1, max_dream_length+1, (len(states),))
            for i in range(max_dream_length):
                actions = torch.randint(0, bit_length, (len(goals),))
                goal_features = dynamics(actions[:,None], goal_features)[:,0]
                goals = flip_bits(goals, actions)
                # goal_features = torch.where(i < lengths[:,None], dynamics(actions[:,None], goal_features)[:,0], goal_features)
                # goals = torch.where(i < lengths[:,None], flip_bits(goals, actions), goals)

        finished = torch.zeros(len(features)).bool()
        for i in range(max_dream_length):
            distances = agent(features, goal_features)
            actions = torch.where(
                torch.rand((len(states),)) > epsilon,
                torch.argmin(distances, dim=1),
                torch.randint(0, bit_length, (len(states),)))
            states = flip_bits(states, actions)
            finished = finished | torch.all(states == goals, dim=1)
            with torch.no_grad():
                features = dynamics(actions[:,None], features)[:,0]
                best_future_distances = torch.clip(agent(features, goal_features).min(dim=1).values * ~finished, 0, max_dream_length+1)
            loss = F.smooth_l1_loss(distances[torch.arange(len(actions)), actions], best_future_distances + 1)
            loss.backward()

        agent_optimizer.step()
        agent_optimizer.zero_grad()
        dream_wins += finished.sum()
        dreams += len(finished)

    # Train the dynamics model
    if len(dynamics_replay_buffer) > dynamics_bs:
        if rolling_dynamics_accuracy < .999:
              dynamics_updates = 8
        elif np.random.uniform() < 0.01:
              dynamics_updates = 1
        else: dynamics_updates = 0

        for _ in range(dynamics_updates):
            states, actions, next_states, mask = dynamics_replay_buffer.sample(dynamics_bs)
            features = encoder(states)
            next_features = encoder(next_states)
            next_feature_preds = dynamics(actions, features[:,0])
            state_preds = decoder(features)
            next_state_preds = decoder(next_features)

            dynamics_loss = F.mse_loss(next_feature_preds[mask], next_features[mask])
            decoder_loss = F.mse_loss(state_preds[mask], states[mask].float())
            loss = dynamics_loss + decoder_loss
            loss.backward()
            dynamics_optimizer.step()
            dynamics_optimizer.zero_grad()

            accuracy = (next_state_preds.detach().round() == next_states).float().mean()
            rolling_dynamics_accuracy = .95 * rolling_dynamics_accuracy + .05 * accuracy
            decoder_errors.append(decoder_loss.item())
            dynamics_errors.append(dynamics_loss.item())
            dynamics_accuracies.append(accuracy)

    # Report stats every so often
    if episode_counter % report_every == 0:
        print(f"Episode {episode_counter:<5} | "
              f"Epsilon: {epsilon:<4.2f} | "
              f"Real Wins: {real_wins:>4} / {report_every} | "
              f"Dream Wins: {dream_wins:>4} / {dreams} | "
              f"Avg Episode Length: {np.mean(episode_lengths):.2f} | "
              f"Decoder error: {np.mean(decoder_errors):<6.4f} | "
              f"Dynamics error: {np.mean(dynamics_errors):<6.4f} | "
              f"Dynamics accuracy: {np.mean(dynamics_accuracies)*100:>6.2f}% | "
              f"Time Taken: {time.time() - last_report:.2f}s")
        real_wins, dream_wins, dreams, episode_lengths = 0, 0, 0, []
        decoder_errors, dynamics_errors, dynamics_accuracies = [], [], []
        last_report = time.time()
