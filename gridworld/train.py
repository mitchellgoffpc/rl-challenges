#!/usr/bin/env python3
import torch
import random
import argparse
from itertools import count
from torch.distributions.categorical import Categorical

from helpers import Episode, ReplayMemory
from gridworld.agent import GridworldAgent
from gridworld.environment import GridworldEnvironment

OUTPUT_SIZE = 2


def parse_args():
    parser = argparse.ArgumentParser(description="RL agent for solving the gridworld environment")

    # Task parameters
    parser.add_argument("--grid_size", type=int, default=5, help="Iterations between reports")
    parser.add_argument("--report-interval", type=int, default=100, help="Iterations between reports")

    # Training parameters
    parser.add_argument("--num-episodes", type=int, default=4000, help="Number of episodes to train for")
    parser.add_argument('--max-episode-length', type=int, default=30, help="Maximum number of steps per episode")
    parser.add_argument('--num-dreams', type=int, default=0, help="Number of 'dream' episodes to generate after each real episode")
    parser.add_argument('--max-dream-length', type=int, default=5, help="Maximum number of steps per dream before encountering a reward")
    parser.add_argument("--memory-size", type=int, default=8000, help="Maximum number of transitions to store in the replay memory")
    parser.add_argument("--batch-size", type=int, default=512, help="Number of transitions to sample per mini-batch")
    parser.add_argument("--hidden-layer-size", type=int, default=128, help="Width of the agent's hidden layer")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Decay factor for rewards")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Clip factor for policy gradients")

    return parser.parse_args()


# Training

def train(flags):
    agent = GridworldAgent(flags)
    old_agent = GridworldAgent(flags)
    env = GridworldEnvironment(flags.grid_size, flags.grid_size)
    optimizer = torch.optim.Adam(agent.parameters(), lr=flags.learning_rate)

    episode = Episode()
    dream_episode = Episode()
    memory = ReplayMemory(flags.memory_size)
    episode_lengths, total_wins = [], 0

    for episode_counter in range(1, flags.num_episodes + 1):
        state, goal = env.reset()
        state, goal = torch.from_numpy(state), torch.from_numpy(goal)

        # Run an episode
        for step in range(flags.max_episode_length):
            with torch.no_grad():
                policy = agent(state[None], goal[None])
            dist = Categorical(policy)
            action = dist.sample()

            next_state, _, reward, done = env.step(action.item())
            episode.push(state, action, torch.tensor(reward), torch.tensor(next_state), goal)
            state = torch.from_numpy(next_state)

            if done:
                if reward == 1: # We won!
                    total_wins += 1
                break

        # Create some short 'dream' episodes where the goal state is one we actually encountered in the last episode.
        for i in torch.randint(0, len(episode), (flags.num_dreams,)):
            state, action, _, goal, _ = episode.memory[i]
            dream_episode.push(state, action, torch.tensor(1.), goal, goal)

            start = i - torch.randint(0, flags.max_dream_length, ())
            for j in range(max(0, start), i):
                state, action, _, next_state, _ = episode.memory[j]
                dream_episode.push(state, action, torch.tensor(0.), next_state, goal)

            dream_episode.discount_rewards(flags.gamma)
            memory.push_episode(dream_episode)

        # When the episode finishes, reset the environment and update the agent
        episode_lengths.append(step + 1.)
        episode.discount_rewards(flags.gamma)
        memory.push_episode(episode)

        if len(memory) > flags.batch_size * 8:
            optimizer.zero_grad()
            old_agent.load_state_dict(agent.state_dict())

            inputs, actions, rewards, _, goals = memory.sample(flags.batch_size, normalize=True)
            outputs, old_outputs = agent(inputs, goals), old_agent(inputs, goals)
            responsible_outputs = torch.gather(outputs, 1, actions)
            old_responsible_outputs = torch.gather(old_outputs, 1, actions).detach()

            dist = Categorical(outputs)
            ratio = responsible_outputs / (old_responsible_outputs + 1e-5)
            clamped_ratio = torch.clamp(ratio, 1. - flags.epsilon, 1. + flags.epsilon)
            loss = -torch.min(ratio * rewards, clamped_ratio * rewards).mean()

            loss.backward()
            optimizer.step()

        # Report stats every so often
        if episode_counter % flags.report_interval == 0:
            report = "Episode {:<5} | Wins: {:>3} / {} | Avg Episode Length: {:.2f}"
            print(report.format(episode_counter, total_wins, flags.report_interval, torch.tensor(episode_lengths).mean()))
            episode_lengths, total_wins = [], 0


if __name__ == '__main__':
    train(parse_args())
