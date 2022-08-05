import time
import torch
import torch.nn.functional as F
import numpy as np
from helpers import ReplayBuffer, mean, get_contrastive_loss
from maze.models import MazeAgent, Encoder, Decoder, Dynamics, NUM_ACTIONS
from maze.environment import MazeEnvironment


grid_w, grid_h = 5,5
agent_hidden_size = 256
dynamics_hidden_size = 128
agent_batch_size = 512
dynamics_batch_size = 64
num_episodes = 20000
max_episode_length = 20
max_dream_length = 5
report_every = 100

agent = MazeAgent(agent_hidden_size, dynamics_hidden_size)
encoder = Encoder(dynamics_hidden_size, grid_w, grid_h)
decoder = Decoder(dynamics_hidden_size, grid_w, grid_h)
dynamics = Dynamics(dynamics_hidden_size)
logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
agent_optimizer = torch.optim.Adam(agent.parameters(), lr=0.0003)
dynamics_optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters(), *dynamics.parameters(), logit_scale], lr=0.0003)

env = MazeEnvironment(grid_w, grid_h)
agent_replay_buffer = ReplayBuffer(32000)
dynamics_replay_buffer = ReplayBuffer(1024)
real_wins, dream_wins, dreams, episode_lengths = 0, 0, 0, []
decoder_errors, dynamics_errors, dynamics_accuracies = [], [], []
rolling_dynamics_error = 1.0
dream_success_rate = 0.0
last_report = time.time()

def env_step(state, action):
    y,x = np.nonzero(state[:,:,-1])[0]
    if env.walls[y,x,action]:
        return state
    env.position = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)][action]
    return env.get_observation()


for episode_counter in range(1, num_episodes + 1):
    epsilon = 0.1 # max(0.00, 1. - 2 * float(episode_counter) / num_episodes)

    # Play an episode
    episode = []
    with torch.no_grad():
        state = env.reset()
        goal = env.get_observation(env.target)
        goal_features = encoder(goal[None])

    for step in range(max_episode_length):
        with torch.no_grad():
            distances = agent(encoder(state[None]), goal_features)

        # Epsilon-greedy exploration strategy
        if torch.rand(()) > epsilon:
              action = torch.argmin(distances)
        else: action = torch.randint(0, NUM_ACTIONS, ())

        next_state, reward, done, _ = env.step(action)
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


    # # Train the agent on real episodes
    # if len(agent_replay_buffer) > agent_batch_size:
    #     states, goals, actions, next_states, finished = agent_replay_buffer.sample(agent_batch_size)
    #
    #     with torch.no_grad():
    #         features, next_features, goal_features = encoder(states), encoder(next_states), encoder(goals)
    #         best_future_distances = torch.clip(agent(next_features, goal_features).min(dim=1).values * ~finished, 0, max_episode_length+1)
    #     distances = agent(features, goal_features)[torch.arange(len(actions)), actions]
    #     loss = F.smooth_l1_loss(distances, best_future_distances + 1)
    #     loss.backward()
    #
    #     agent_optimizer.step()
    #     agent_optimizer.zero_grad()

    # Train the agent on dream episodes
    if len(agent_replay_buffer) > agent_batch_size:
        states, _, _, _, _ = agent_replay_buffer.sample(agent_batch_size)
        with torch.no_grad():
            features = encoder(states)

        # Single-step
        with torch.no_grad():
            actions = torch.randint(0, NUM_ACTIONS, (len(states),))
            goal_features = dynamics(actions[:,None], features)[:,0]
        distances = agent(features, goal_features)
        loss = F.smooth_l1_loss(distances[torch.arange(len(actions)), actions], torch.ones(len(actions)))
        loss.backward()

        # Multi-step
        with torch.no_grad():
            goal_indices = torch.randint(0, len(states), size=(len(states), 8))
            goals, goal_features = states[goal_indices], features[goal_indices]
            b,n,f = goal_features.shape
            distances = agent(features[:,None].repeat(1,8,1).view(b*n,f), goal_features.view(b*n,f)).view(b,n,-1).min(dim=2).values
            probs = F.softmax(1 / (1 + torch.abs(.8*max_dream_length - distances)), dim=1)
            goal_indices = torch.argmax((probs.cumsum(dim=1) > torch.rand(len(states), 1)).int(), dim=1)
            goal_features = goal_features[torch.arange(len(states)), goal_indices]
            goals = goals[torch.arange(len(states)), goal_indices]

            # mask = torch.rand(len(goals)) > 0.5
            # for i in range(max_dream_length * 3):
            #     actions = torch.randint(0, NUM_ACTIONS, (len(goals),))
            #     # best_actions = torch.argmax(agent(goal_features, features), dim=1)
            #     # actions = torch.where(mask, rand_actions, best_actions)
            #     # goal_features = dynamics(actions[:,None], goal_features)[:,0]
            #     goals = torch.as_tensor(np.stack([env_step(goals[j], actions[j]) for j in range(agent_batch_size)], axis=0))
            # goal_features = encoder(goals)

            # goals = []
            # import random
            # for _ in range(agent_batch_size):
            #     env.position = (random.randrange(3), random.randrange(3))
            #     goals.append(torch.as_tensor(env.get_observation()))
            # goals = torch.stack(goals)
            # goal_features = encoder(goals)

        finished = torch.zeros(len(features)).bool()
        for i in range(max_dream_length):
            distances = agent(features, goal_features)
            actions = torch.where(
                torch.rand((len(states),)) > epsilon,
                torch.argmin(distances, dim=1),
                torch.randint(0, NUM_ACTIONS, (len(states),)))
            new_states = torch.as_tensor(np.stack([env_step(states[j], actions[j]) for j in range(agent_batch_size)], axis=0))
            states = torch.where(finished[:,None,None,None], states, new_states)
            finished = finished | torch.all(states.flatten(start_dim=1) == goals.flatten(start_dim=1), dim=1)
            with torch.no_grad():
                features = torch.where(finished[:,None], features, encoder(states))
                best_future_distances = torch.clip(agent(features, goal_features).min(dim=1).values * ~finished, 0, max_dream_length+1)
            loss = F.smooth_l1_loss(distances[torch.arange(len(actions)), actions], best_future_distances + 1)
            loss.backward()

        agent_optimizer.step()
        agent_optimizer.zero_grad()
        dream_wins += finished.sum()
        dreams += agent_batch_size
        dream_success_rate = 0.998 * dream_success_rate + 0.002 * (dream_wins / dreams)
        if dream_success_rate > .4 and max_dream_length < max_episode_length:
            print(f"Bumping max_dream_length to {max_dream_length + 1}")
            dream_success_rate = 0.0
            max_dream_length += 1

    # Train the dynamics model
    if len(dynamics_replay_buffer) > dynamics_batch_size:
        # if rolling_dynamics_error > .0001:
        if episode_counter < 1000:
              dynamics_updates = 8
        else: dynamics_updates = 0

        for _ in range(dynamics_updates):
            states, actions, next_states, mask = dynamics_replay_buffer.sample(dynamics_batch_size)
            features = encoder(states)
            next_features = encoder(next_states)
            next_feature_preds = dynamics(actions, features[:,0])
            state_preds = decoder(features.detach())
            next_state_preds = decoder(next_features.detach())

            bs, ts, _ = features.shape
            i = torch.randint(0, ts, (bs,))
            contrastive_loss = get_contrastive_loss(features[torch.arange(bs),i], next_features[torch.arange(bs),i], logit_scale)
            dynamics_loss = F.mse_loss(next_feature_preds[mask], next_features[mask])
            decoder_loss = F.mse_loss(state_preds[mask], states[mask].float()) + F.mse_loss(next_state_preds[mask], next_states[mask].float())

            loss = dynamics_loss + decoder_loss + contrastive_loss
            loss.backward()
            dynamics_optimizer.step()
            dynamics_optimizer.zero_grad()

            accuracy = (next_state_preds.detach().round() == next_states).float().mean()
            rolling_dynamics_error = .95 * rolling_dynamics_error + .05 * dynamics_loss.item()
            decoder_errors.append(decoder_loss.item())
            dynamics_errors.append(dynamics_loss.item())
            dynamics_accuracies.append(accuracy)

    # Report stats every so often
    if episode_counter % report_every == 0:
        print(f"Episode {episode_counter:<5} | "
              f"Epsilon: {epsilon:<4.2f} | "
              f"Real Wins: {real_wins:>4} / {report_every} | "
              f"Dream Wins: {dream_wins:>4} / {dreams} | "
              f"Avg Episode Length: {mean(episode_lengths):.2f} | "
              f"Decoder error: {mean(decoder_errors):<6.5f} | "
              f"Dynamics error: {mean(dynamics_errors):<6.5f} | "
              f"Dynamics accuracy: {mean(dynamics_accuracies)*100:>6.2f}% | "
              f"Time Taken: {time.time() - last_report:.2f}s")
        real_wins, dream_wins, dreams, episode_lengths = 0, 0, 0, []
        decoder_errors, dynamics_errors, dynamics_accuracies = [], [], []
        last_report = time.time()
