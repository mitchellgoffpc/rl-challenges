import time
import torch
import torch.nn.functional as F
from helpers import ReplayBuffer, mean
from bitflip.models import BitflipAgent, Encoder, Decoder, Dynamics
from bitflip.environment import build_example, flip_bits


bit_length = 8
agent_hidden_size = 128
dynamics_hidden_size = 64
agent_batch_size = 512
dynamics_batch_size = 128
num_episodes = 20000
max_episode_length = bit_length * 3//2
max_dream_length = 3
report_every = 100

agent = BitflipAgent(agent_hidden_size, dynamics_hidden_size, bit_length)
encoder = Encoder(dynamics_hidden_size, bit_length)
decoder = Decoder(dynamics_hidden_size, bit_length)
dynamics = Dynamics(dynamics_hidden_size, bit_length)
logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
agent_optimizer = torch.optim.Adam(agent.parameters(), lr=0.0003)
dynamics_optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters(), *dynamics.parameters(), logit_scale], lr=0.0003)

agent_replay_buffer = ReplayBuffer(32000)
dynamics_replay_buffer = ReplayBuffer(1024)
real_wins, dream_wins, dreams, episode_lengths = 0, 0, 0, []
decoder_errors, dynamics_errors, dynamics_accuracies = [], [], []
rolling_dynamics_error = 1.0
dream_success_rate = 0.0
last_report = time.time()


for episode_counter in range(1, num_episodes + 1):
    epsilon = max(0.00, 1. - 3 * float(episode_counter) / num_episodes)

    # Play an episode
    episode = []
    with torch.no_grad():
        state, goal = build_example(bit_length), build_example(bit_length)
        goal_features = encoder(goal[None])

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
    # if len(agent_replay_buffer) > agent_batch_size:
    #     states, goals, actions, next_states, finished = agent_replay_buffer.sample(agent_batch_size)
    #
    #     with torch.no_grad():
    #         features, next_features, goal_features = encoder(states), encoder(next_states), encoder(goals)
    #         best_future_distances = torch.clip(agent(next_features, goal_features).min(dim=1).values * ~finished, 0, bit_length)
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
            actions = torch.randint(0, bit_length, (len(states),))
            goal_features = dynamics(actions[:,None], features)[:,0]
        distances = agent(features, goal_features)
        loss = F.smooth_l1_loss(distances[torch.arange(len(actions)), actions], torch.ones(len(actions)))
        loss.backward()

        # Multi-step
        with torch.no_grad():
            goals, goal_features = states, features
            for i in range(max_dream_length):
                actions = torch.randint(0, bit_length, (len(goals),))
                goal_features = dynamics(actions[:,None], goal_features)[:,0]
                goals = flip_bits(goals, actions)

        finished = torch.zeros(len(features)).bool()
        for i in range(max_dream_length):
            distances = agent(features, goal_features)
            actions = torch.where(
                torch.rand((len(states),)) > epsilon,
                torch.argmin(distances, dim=1),
                torch.randint(0, bit_length, (len(states),)))
            states = torch.where(finished[:,None], states, flip_bits(states, actions))
            finished = finished | torch.all(states == goals, dim=1)
            with torch.no_grad():
                features = torch.where(finished[:,None], features, dynamics(actions[:,None], features)[:,0])
                best_future_distances = torch.clip(agent(features, goal_features).min(dim=1).values * ~finished, 0, max_dream_length+1)
            loss = F.smooth_l1_loss(distances[torch.arange(len(actions)), actions], best_future_distances + 1)
            loss.backward()

        agent_optimizer.step()
        agent_optimizer.zero_grad()
        dream_wins += finished.sum()
        dreams += len(finished)
        dream_success_rate = 0.99 * dream_success_rate + 0.01 * (dream_wins / dreams)
        # if dream_success_rate > .95 and max_dream_length < bit_length // 2:
        #     print(f"Bumping max_dream_length to {max_dream_length + 1}")
        #     dream_success_rate = 0.0
        #     max_dream_length += 1

    # Train the dynamics model
    if len(dynamics_replay_buffer) > dynamics_batch_size:
        if rolling_dynamics_error > .0001:
              dynamics_updates = 8
        elif rolling_dynamics_error > .00002:
              dynamics_updates = 1
        else: dynamics_updates = 0

        for _ in range(dynamics_updates):
            states, actions, next_states, mask = dynamics_replay_buffer.sample(dynamics_batch_size)
            features = encoder(states)
            next_features = encoder(next_states)
            next_feature_preds = dynamics(actions, features[:,0])
            state_preds = decoder(features.detach())
            next_state_preds = decoder(next_features.detach())

            dynamics_loss = F.mse_loss(next_feature_preds[mask], next_features[mask])
            decoder_loss = F.mse_loss(state_preds[mask], states[mask].float()) + F.mse_loss(next_state_preds[mask], next_states[mask].float())

            i = torch.randint(0, 12, (len(features),))
            feats, nfeats = features[torch.arange(len(features)),i], next_features[torch.arange(len(features)),i]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            nfeats = nfeats / nfeats.norm(dim=-1, keepdim=True)
            logits = logit_scale.exp() * feats @ nfeats.t()
            targets = torch.arange(0, len(feats)).long()
            contrastive_loss = F.cross_entropy(logits, targets)
            # contrastive_loss = (F.nll_loss(F.log_softmax(logits, dim=0), targets) + F.nll_loss(F.log_softmax(logits, dim=1), targets)) / 2

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
