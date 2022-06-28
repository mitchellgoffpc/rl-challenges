import gym
import time
import torch
import torch.nn.functional as F
from helpers import InfiniteReplayBuffer, mean
from cartpole.models import OUTPUT_SIZE, Encoder, Decoder, Dynamics, Value


value_hidden_size = 64
dynamics_hidden_size = 64
value_batch_size = 512
dynamics_batch_size = 32
value_train_steps = 4
dynamics_train_steps = 32
temporal_len = 16
report_every = 25
gamma = 0.8

value = Value(value_hidden_size, dynamics_hidden_size)
encoder = Encoder(dynamics_hidden_size)
decoder = Decoder(dynamics_hidden_size)
dynamics = Dynamics(dynamics_hidden_size)
value_optimizer = torch.optim.Adam(value.parameters(), lr=0.0003)
dynamics_optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters(), *dynamics.parameters()], lr=0.0003)

env = gym.make('CartPole-v1')
replay_buffer = InfiniteReplayBuffer(32000)
decoder_errors, dynamics_errors, reward_errors, dynamics_done_preds, value_done_preds = [], [], [], [], []
last_report = time.time()


for episode_counter in range(2000):
    done = False
    old_state = torch.as_tensor(env.reset())
    for step in range(500):
        action = torch.randint(0, OUTPUT_SIZE, ())
        new_state, reward, done, _ = env.step(action.item())
        new_state = torch.as_tensor(new_state)
        reward = torch.tensor(-1 if done and step < 499 else 0)
        replay_buffer.push(old_state, action, new_state, reward)
        old_state = new_state
        if done: break

    # Update value network
    if len(replay_buffer) > value_batch_size * 4:
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
              f"Dynamics error: {mean(dynamics_errors):.4f} | "
              f"Decoder error: {mean(decoder_errors):.4f} | "
              f"Reward error: {mean(reward_errors):.4f} | "
              f"Dynamics done preds: {mean(dynamics_done_preds):.4f} | "
              f"Value done preds: {mean(value_done_preds):.4f} | "
              f"Time Taken: {time.time() - last_report:.2f}s")
        decoder_errors, dynamics_errors, reward_errors, dynamics_done_preds, value_done_preds = [], [], [], [], []
        last_report = time.time()
