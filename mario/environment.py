import torch
from nes import NES

class MarioEnvironment:
  def __init__(self):
    self.nes = NES("mario.nes", headless=True, verbose=False, sync_mode=0)
    self.nes.run_frame_headless(run_frames=60)
    self.nes.run_frame_headless(run_frames=180, controller1_state=[0,0,0,1,0,0,0,0])

  def step(self, action):
    return torch.as_tensor(self.nes.run_frame_headless(run_frames=1, controller1_state=action.tolist())).permute(2,0,1)
