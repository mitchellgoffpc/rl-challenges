import numpy as np


class GridworldEnvironment:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
        self.position = np.random.randint(self.height), np.random.randint(self.width)
        self.target = np.random.randint(self.height), np.random.randint(self.width)
        return self.render()

    def step(self, action):
        row, col = self.position
        if action == 0: # UP
            self.position = max(0, row-1), col
        elif action == 1: # DOWN
            self.position = min(self.height-1, row+1), col
        elif action == 2: # LEFT
            self.position = row, max(0, col-1)
        elif action == 3: # RIGHT
            self.position = row, min(self.width-1, col+1)

        obs, target = self.render()
        finished = self.finished()
        reward = 1 if finished else 0
        return obs, target, reward, finished

    def render(self):
        obs = self.grid.copy()
        obs[self.position] = 1
        target = self.grid.copy()
        target[self.target] = 1
        return obs, target

    def finished(self):
        return self.position == self.target
