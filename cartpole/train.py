#!/usr/bin/env python3
import gym
import torch
import argparse
from itertools import count
from torch.distributions.categorical import Categorical

from helpers import Episode, ReplayMemory
from cartpole.agent import CartPoleAgent


def parse_args():
    parser = argparse.ArgumentParser(description="RL agent for solving the cartpole environment")

    # Task parameters
    parser.add_argument("--report-interval", type=int, default=100, help="Iterations between reports")
    parser.add_argument("--render-interval", type=int, default=0, help="Iterations between rendering episodes of the game")

    # Training parameters
    parser.add_argument("--num-episodes", type=int, default=2000, help="Number of episodes to train for")
    parser.add_argument("--memory-size", type=int, default=32000, help="Maximum number of transitions to store in the replay memory")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of transitions to sample per mini-batch")
    parser.add_argument("--hidden-layer-size", type=int, default=128, help="Width of the agent's hidden layer")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Optimizer learning rate")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Clip factor for policy improvement")
    parser.add_argument("--gamma", type=float, default=0.9, help="Decay factor for rewards")

    return parser.parse_args()


# Training

def train(flags):
    env = gym.make('CartPole-v1')
    agent = CartPoleAgent(flags)
    old_agent = CartPoleAgent(flags)
    optimizer = torch.optim.Adam(agent.parameters(), lr=flags.learning_rate)

    episode = Episode()
    memory = ReplayMemory(flags.memory_size)
    episode_lengths, avg_episode_length = [], 0

    for episode_counter in range(1, flags.num_episodes + 1):
        state = env.reset()
        state = torch.from_numpy(state)

        # Run an episode
        for step in count():
            with torch.no_grad():
                policy = agent(state.view(1, -1))
            dist = Categorical(policy)
            action = dist.sample()

            next_state, reward, done, stats = env.step(action.item())
            episode.push(state, action, reward)
            state = torch.from_numpy(next_state)

            if flags.render_interval and episode_counter % flags.render_interval == 0:
                env.render()
            if done: break

        # When the episode finishes, reset the environment and update the agent
        episode_lengths.append(step + 1.)
        episode.discount_rewards(flags.gamma)
        memory.push_episode(episode)

        if len(memory) > flags.batch_size * 4 and avg_episode_length < 500:
            optimizer.zero_grad()
            old_agent.load_state_dict(agent.state_dict())

            inputs, actions, rewards = memory.sample(flags.batch_size, normalize=True)
            outputs, old_outputs = agent(inputs), old_agent(inputs)
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
            avg_episode_length = torch.tensor(episode_lengths).mean()
            print("Episode {:<4} | Avg episode length: {:.2f}".format(episode_counter, avg_episode_length))
            episode_lengths = []


if __name__ == '__main__':
    train(parse_args())
