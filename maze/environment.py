import random
import torch
import numpy as np

def make_maze(w, h):
    visited = np.ones((h+1, w+1), dtype=bool)
    visited[:h, :w] = False
    walls = np.ones((h, w, 4), dtype=bool)

    def walk(x, y):
        visited[y, x] = True
        directions = [(0, x, y-1), (1, x+1, y), (2, x, y+1), (3, x-1, y)]
        random.shuffle(directions)
        for i, nx, ny in directions:
            if visited[ny, nx]: continue
            walls[y, x, i] = False
            walls[ny, nx, i-2] = False
            walk(nx, ny)

    walk(random.randrange(w), random.randrange(h))
    return walls[:h, :w]

def maze_to_string(walls):
    h, w, _ = walls.shape
    s = ''
    for y in range(h):
        s += ''.join('+--' if walls[y,x,0] else '+  ' for x in range(w)) + '+\n'
        s += ''.join('|  ' if walls[y,x,3] else '   ' for x in range(w)) + '|\n'
    return s + ''.join('+--' for _ in range(w)) + '+\n'


class MazeEnvironment:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.walls = make_maze(w, h)
        self.start = (random.randrange(self.w), random.randrange(self.h))
        self.target = (random.randrange(self.w), random.randrange(self.h))
        self.reset()

    def reset(self):
        self.position = self.start
        return self.get_observation()

    def step(self, action):
        x, y = self.position
        if self.walls[y,x,action]:
            return self.get_observation(), 0, False, self.get_info()
        self.position = [(x, y-1), (x+1, y), (x, y+1), (x-1, y)][action]
        if self.position == self.target:
            return self.get_observation(), 1, True, self.get_info()
        return self.get_observation(), 0, False, self.get_info()

    def get_observation(self):
        x, y = self.position
        position_obs = np.zeros((self.h, self.w, 1), dtype=bool)
        position_obs[y, x] = True
        return torch.as_tensor(np.concatenate([self.walls, position_obs], axis=-1))

    def get_info(self):
        return {'position': self.position}
