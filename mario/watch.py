import sys
import cv2
import numpy as np
from pathlib import Path

if len(sys.argv) > 1:
    data = np.load(str(Path(__file__).parent / f'episodes/{sys.argv[1]}.npz'))['arr_0']
else:
    import random
    from mario.environment import MarioEnv
    # from mario.environment import ZeldaEnv as MarioEnv
    env = MarioEnv()
    env.reset()
    data = []
    for step in range(60):
        action, = random.choices([0, env.A, env.LEFT, env.RIGHT, env.LEFT | env.A, env.RIGHT | env.A], weights=[1, 1, 1, 10, 1, 5])
        # action = random.choice([0, env.A, env.UP, env.DOWN, env.LEFT, env.RIGHT])
        for _ in range(4):
            frame, reward, done, _ = env.step(action)
        data.append(frame)
    data = np.array(data)

out = cv2.VideoWriter(str(Path(__file__).parent / 'output.mp4'), cv2.VideoWriter_fourcc(*'avc1'), 20.0, (data.shape[2], data.shape[1]))
for frame in data:
    out.write(frame[:,:,::-1])
out.release()
