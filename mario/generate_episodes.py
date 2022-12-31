import random
import numpy as np
import multiprocessing
from tqdm import tqdm

def generate_episode(i):
    from mario.environment import MarioEnv
    env = MarioEnv()
    env.reset()
    episode = []
    random.seed(i)
    for step in range(60):
        action = random.choice([0, 1, 1<<6, 1<<7, (1<<6)+1, (1<<7)+1])
        for _ in range(4):
            frame, reward, done, _ = env.step(action)
        episode.append(frame)
        if reward or done:
            print(reward, done)
    np.savez_compressed(f'./mario/episodes/{i:05d}.npz', np.array(episode))

if __name__ == '__main__':
    with multiprocessing.Pool(8) as pool:
        results = list(tqdm(pool.imap(generate_episode, list(range(1000)))))
