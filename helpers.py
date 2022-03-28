import torch
import random


class Episode:
    memory = []

    def reset(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(list(args))

    def stack(self):
        return map(torch.stack, zip(*self.memory))

    def discount_rewards(self, gamma, normalize = False):
        running_add = 0.
        total_reward = 0.
        for i, (_, _, reward, *_) in list(enumerate(self.memory))[::-1]:
            running_add = running_add * gamma + reward
            total_reward += running_add

        running_add = 0.
        avg_reward = total_reward / len(self.memory) if normalize else 0.
        for i, (state, action, reward, *rest) in list(enumerate(self.memory))[::-1]:
            running_add = running_add * gamma + reward
            discounted_reward = torch.tensor([running_add - avg_reward])
            self.memory[i] = (state, action, discounted_reward, *rest)

    def __len__(self):
        return len(self.memory)


class ReplayMemory:
    memory = []

    def __init__(self, capacity = None):
        self.capacity = capacity

    def reset(self):
        self.memory = []

    def push(self, *args):
        self.push_transition(tuple(args))

    def push_episode(self, episode):
        for step in episode.memory:
            self.push_transition(step)
        episode.reset()

    def push_transition(self, transition):
        if self.capacity is None or len(self.memory) < self.capacity:
              self.memory.append(transition)
        else: self.memory[random.randint(0, len(self.memory) - 1)] = transition

    def sample(self, batch_size, normalize = False):
        sample = random.sample(self.memory, batch_size)
        batch = list(map(torch.stack, zip(*sample)))
        return self.normalize(batch) if normalize else batch

    def normalize(self, batch):
        inputs, states, rewards, *rest = batch
        return (inputs, states, rewards - sum(rewards) / len(rewards), *rest)

    def __len__(self):
        return len(self.memory)


class ReplayBuffer:
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
        sample = random.sample(range(len(self)), batch_size)
        return [torch.stack([self.samples[i][j] for i in sample]) for j in range(len(self.samples[0]))]

    def __len__(self):
        return len(self.samples)
