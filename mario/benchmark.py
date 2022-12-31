from tqdm import trange
import multiprocessing

def run():
  from nes import NES
  nes = NES("roms/mario.nes", headless=True, verbose=False)
  frame = nes.run_frame_headless(run_frames=60)
  frame = nes.run_frame_headless(run_frames=180, controller1_state=[0,0,0,1,0,0,0,0])

  keys = [0] * 7 + [1]
  for _ in trange(10000):
    frame = nes.run_frame_headless(run_frames=1, controller1_state=keys)

def run():
  from nes_py import NESEnv
  env = NESEnv("roms/mario.nes")
  env.reset()
  for _ in range(60):
    frame = env.step(0)
  for _ in range(180):
    frame = env.step(0)
  for _ in trange(10000):
    frame = env.step(0)

if __name__ == '__main__':
  multiprocessing.freeze_support()
  children = [multiprocessing.Process(target=run) for _ in range(8)]
  for p in children:
    p.start()
  for p in children:
    p.join()
