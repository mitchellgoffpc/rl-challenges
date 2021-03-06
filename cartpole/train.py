import gym
import time
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from helpers import InfiniteReplayBuffer, mean
from cartpole.models import OUTPUT_SIZE, Distance, Value, Encoder, Decoder, Dynamics


value_hidden_size = 64
dynamics_hidden_size = 64
value_batch_size = 512
dynamics_batch_size = 32
value_train_steps = 4
dynamics_train_steps = 32
max_dream_length = 1
temporal_len = 16
report_every = 25
gamma = 0.8

distance = Distance(value_hidden_size, dynamics_hidden_size)
value = Value(value_hidden_size, dynamics_hidden_size)
encoder = Encoder(dynamics_hidden_size)
decoder = Decoder(dynamics_hidden_size)
dynamics = Dynamics(dynamics_hidden_size)
distance_optimizer = torch.optim.Adam(distance.parameters(), lr=0.0003)
value_optimizer = torch.optim.Adam(value.parameters(), lr=0.0003)
dynamics_optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters(), *dynamics.parameters()], lr=0.0003)

env = gym.make('CartPole-v1')
replay_buffer = InfiniteReplayBuffer(32000)
decoder_errors, dynamics_errors, reward_errors, dynamics_done_preds, value_done_preds = [], [], [], [], []
real_wins, dream_wins, dreams, episode_lengths = 0, 0, 0, []
last_report = time.time()


for episode_counter in range(2000):
    epsilon = max(0.1, 1. - episode_counter / 1000)

    # Play an episode
    done = False
    old_state = torch.as_tensor(env.reset())
    for step in range(500):
        action = torch.randint(0, OUTPUT_SIZE, ())
        # with torch.no_grad():
        #     values = value(old_state[None])
        #
        # # Epsilon-greedy exploration strategy
        # if torch.rand(()) > epsilon:
        #       action = torch.argmax(values)
        # else: action = torch.randint(0, OUTPUT_SIZE, ())

        new_state, _, done, _ = env.step(action.item())
        new_state = torch.as_tensor(new_state)
        reward = torch.tensor(-1 if done and step < 499 else 0)
        replay_buffer.push(old_state, action, new_state, reward)
        old_state = new_state
        if step == 499: real_wins += 1
        if done: break
    episode_lengths.append(step)

    # Train the agent on dream episodes
    if len(replay_buffer) > value_batch_size * 4:
        states, _, _, _ = [x.squeeze() for x in replay_buffer.sample(value_batch_size, 1)]
        with torch.no_grad():
            features = encoder(states)

        # Single-step
        with torch.no_grad():
            actions = torch.randint(0, OUTPUT_SIZE, (len(states),))
            goal_features = dynamics(actions[:,None], features)[0][:,0]

        distances = distances0 = distance(features, features)
        loss = F.smooth_l1_loss(distances, torch.zeros_like(distances))
        loss.backward()

        distances = distance(features, goal_features)
        # print(distances[torch.arange(len(actions)), actions].mean(), distances0.mean())
        loss = F.smooth_l1_loss(distances[torch.arange(len(actions)), actions], torch.ones(len(actions)))
        loss.backward()

        distance_optimizer.step()
        distance_optimizer.zero_grad()

        # Multi-step
        with torch.no_grad():
            goal_features = features
            for i in range(max_dream_length):
                actions = torch.randint(0, OUTPUT_SIZE, (len(states),))
                goal_features = dynamics(actions[:,None], goal_features)[0][:,0]

        finished = torch.zeros(len(features)).bool()
        for i in range(max_dream_length):
            distances = distance(features, goal_features)
            actions = torch.argmin(distances, dim=1)
            # print(distances[torch.arange(len(actions)), actions].mean().item())
            # actions = torch.where(
            #     torch.rand((len(states),)) > epsilon,
            #     torch.argmin(distances, dim=1),
            #     torch.randint(0, OUTPUT_SIZE, (len(states),)))
            finished = finished | (distances[torch.arange(len(actions)), actions] <= 1.5)
            with torch.no_grad():
                features = torch.where(finished[:,None], features, dynamics(actions[:,None], features)[0][:,0])
                best_future_distances = torch.clip(distance(features, goal_features).min(dim=1).values * ~finished, 0, max_dream_length+5)
                print(distances.min(dim=1).values.mean(), best_future_distances.mean())
            loss = F.smooth_l1_loss(distances[torch.arange(len(actions)), actions], best_future_distances + 1)
            loss.backward()

        distance_optimizer.step()
        distance_optimizer.zero_grad()
        dream_wins += finished.sum()
        dreams += len(finished)
        # dream_success_rate = 0.99 * dream_success_rate + 0.01 * (dream_wins / dreams)

    # Update value network
    if len(replay_buffer) > value_batch_size * 16:
        for _ in range(value_train_steps):
            states, actions, new_states, rewards = [x.squeeze() for x in replay_buffer.sample(value_batch_size, 1)]
            finished = rewards == -1
            with torch.no_grad():
                features, new_features = encoder(states), encoder(new_states)
                best_future_action_values = value(new_features).max(dim=1).values * ~finished
            action_values = torch.gather(value(features), dim=1, index=actions.unsqueeze(1)).flatten()
            loss = F.smooth_l1_loss(action_values, rewards + gamma * best_future_action_values)
            loss.backward()
            value_optimizer.step()
            value_optimizer.zero_grad()
            value_done_preds.extend(action_values[rewards == -1])

    # Update dynamics network
    if len(replay_buffer) > dynamics_batch_size * 4:
        for _ in range(dynamics_train_steps):
            old_states, actions, new_states, rewards = replay_buffer.sample(dynamics_batch_size, temporal_len)
            old_features = encoder(old_states)
            new_features = encoder(new_states)
            new_feature_preds, reward_preds = dynamics(actions, old_features[:,0])
            old_state_preds = decoder(old_features)
            new_state_preds = decoder(new_features)

            dynamics_loss = F.mse_loss(new_feature_preds, new_features)
            decoder_loss = F.mse_loss(old_state_preds, old_states)
            reward_loss = F.mse_loss(reward_preds, rewards.float())
            loss = dynamics_loss + decoder_loss + reward_loss
            loss.backward()
            dynamics_optimizer.step()
            dynamics_optimizer.zero_grad()

            dynamics_errors.append(dynamics_loss.item())
            decoder_errors.append(decoder_loss.item())
            reward_errors.append(reward_loss.item())
            dynamics_done_preds.extend(reward_preds[rewards == -1])

    if episode_counter % report_every == 0:
        print(f"Episode {episode_counter:<4} | "
              f"Epsilon: {epsilon:<4.2f} | "
              f"Real Wins: {real_wins:>3} / {report_every} | "
              f"Dream Wins: {dream_wins:>4} / {dreams} | "
              f"Avg Episode Length: {mean(episode_lengths):.2f} | "
              f"Dynamics error: {mean(dynamics_errors):.4f} | "
              f"Decoder error: {mean(decoder_errors):.4f} | "
              f"Reward error: {mean(reward_errors):.4f} | "
              f"Dynamics done preds: {mean(dynamics_done_preds):.4f} | "
              f"Value done preds: {mean(value_done_preds):.4f} | "
              f"Time Taken: {time.time() - last_report:.2f}s")
        decoder_errors, dynamics_errors, reward_errors, dynamics_done_preds, value_done_preds = [], [], [], [], []
        real_wins, dream_wins, dreams, episode_lengths = 0, 0, 0, []
        last_report = time.time()
