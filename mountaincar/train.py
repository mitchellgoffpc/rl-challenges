#!/usr/bin/env python3
import gym
import torch
import random
import argparse
from itertools import count
from torch.distributions.categorical import Categorical

from helpers import Episode, ReplayMemory
from mountaincar.agent import MountainCarAgent

OUTPUT_SIZE = 2


def parse_args():
    parser = argparse.ArgumentParser(description="RL agent for solving the cartpole environment")

    # Task parameters
    parser.add_argument("--report-interval", type=int, default=100, help="Iterations between reports")
    parser.add_argument("--render-interval", type=int, default=0, help="Iterations between rendering episodes of the game")

    # Training parameters
    parser.add_argument("--num-episodes", type=int, default=4000, help="Number of episodes to train for")
    parser.add_argument('--num-dreams', type=int, default=5, help="Number of 'dream' episodes to generate after each real episode")
    parser.add_argument('--max-dream-length', type=int, default=30, help="Maximum number of steps per dream before encountering a reward")
    parser.add_argument("--memory-size", type=int, default=32000, help="Maximum number of transitions to store in the replay memory")
    parser.add_argument("--batch-size", type=int, default=512, help="Number of transitions to sample per mini-batch")
    parser.add_argument("--hidden-layer-size", type=int, default=128, help="Width of the agent's hidden layer")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="Decay factor for rewards")

    return parser.parse_args()


# Training

def train(flags):
    env = gym.make('MountainCar-v0')
    agent = MountainCarAgent(flags)
    optimizer = torch.optim.Adam(agent.parameters(), lr=flags.learning_rate)
    criterion = torch.nn.SmoothL1Loss()

    episode = Episode()
    memory = ReplayMemory(flags.memory_size)
    episode_lengths, total_wins = [], 0

    for episode_counter in range(1, flags.num_episodes + 1):
        goal = torch.tensor([random.uniform(.6, .001), random.uniform(0, .01)])
        state = env.reset()
        state = torch.from_numpy(state)
        epsilon = max(0.05, 1. - float(episode_counter) / 1500)

        # Run an episode
        for step in count():
            with torch.no_grad():
                action_scores = agent(state[None], goal[None])

            # Epsilon-greedy exploration strategy
            if torch.rand(()) > epsilon:
                  action = torch.argmax(action_scores)
            else: action = torch.randint(0, OUTPUT_SIZE, ())

            for _ in range(2):
              next_state, reward, done, stats = env.step(action.item())
            reward = 1 if done and step < 99 else 0
            episode.push(state, action, torch.tensor(reward), torch.tensor(done), torch.from_numpy(next_state), goal)
            state = torch.from_numpy(next_state)

            if flags.render_interval and episode_counter % flags.render_interval == 0:
                env.render()
            if done:
                if reward == 1: # We won!
                    total_wins += 1
                    for s in episode.memory:
                        s[-1] = state # Replace the goal state with the actual state we ended up in
                break

        # Create some short 'dream' episodes where the goal state is one we actually encountered in the last episode.
        for i in torch.randint(0, len(episode), (flags.num_dreams,)):
            state, action, _, _, goal, _ = episode.memory[i]
            memory.push(state, action, torch.tensor(1.), torch.tensor(True), goal, goal)

            start = i - torch.randint(0, flags.max_dream_length, ())
            for j in range(max(0, start), i):
                state, action, _, _, next_state, _ = episode.memory[j]
                memory.push(state, action, torch.tensor(0.), torch.tensor(False), next_state, goal)

        # When the episode finishes, reset the environment and update the agent
        episode_lengths.append(step + 1.)
        memory.push_episode(episode)

        if len(memory) > flags.batch_size * 4:
            optimizer.zero_grad()
            states, actions, rewards, finished, next_states, goals = memory.sample(flags.batch_size)
            with torch.no_grad():
                best_future_action_values = agent(next_states, goals).max(dim=1).values * ~finished
            action_values = torch.gather(agent(states, goals), dim=1, index=actions.unsqueeze(1)).flatten()
            loss = criterion(action_values, rewards + flags.gamma * best_future_action_values)
            loss.backward()
            optimizer.step()

        # Report stats every so often
        if episode_counter % flags.report_interval == 0:
            report = "Episode {:<5} | Epsilon: {:<4.2f} | Wins: {:>3} / {} | Avg Episode Length: {:.2f}"
            print(report.format(episode_counter, epsilon, total_wins, flags.report_interval, torch.tensor(episode_lengths).mean()))
            episode_lengths, total_wins = [], 0


if __name__ == '__main__':
    train(parse_args())
