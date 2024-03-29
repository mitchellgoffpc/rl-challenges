import os
import random
import numpy as np
import multiprocessing
from tqdm import tqdm
from pathlib import Path

def generate_episode(i):
    from mario.environment import MarioEnv
    # from mario.environment import ZeldaEnv as MarioEnv
    env = MarioEnv()
    env.reset()
    episode = []
    random.seed(i)
    for step in range(60):
        action, = random.choices([0, env.A, env.LEFT, env.RIGHT, env.LEFT | env.A, env.RIGHT | env.A], weights=[1, 1, 1, 10, 1, 5])
        # action = random.choice([0, env.A, env.UP, env.DOWN, env.LEFT, env.RIGHT])
        for _ in range(4):
            frame, reward, done, _ = env.step(action)
        episode.append(frame)
    np.savez_compressed(Path(__file__).parent / f'episodes/{i:05d}.npz', np.array(episode))

if __name__ == '__main__':
    (Path(__file__).parent / 'episodes').mkdir(exist_ok=True)
    with multiprocessing.Pool(os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(generate_episode, list(range(1000)))))
