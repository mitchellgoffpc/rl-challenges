import random
import numpy as np
from maze.environment import MazeEnvironment

env = MazeEnvironment(4, 4)

n_explored = []
for _ in range(100):
    env.reset()
    positions = set([])
    for i in range(100):
        _, _, _, info = env.step(random.randrange(4))
        positions.add(info['position'])
    n_explored.append(len(positions))

print(np.mean(n_explored))
