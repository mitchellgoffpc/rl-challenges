import os
import torch
from nes_py import NESEnv

class MarioEnv:
  def __init__(self):
    self.nes = NESEnv(os.path.join(os.path.dirname(__file__), "roms/mario.nes"))

  def reset(self):
    self.nes.reset()
    for _ in range(60):
      self.nes.step(0)
    for _ in range(180):
      frame, *_ = self.nes.step(1 << 3)
    return frame

  def step(self, action):
    return self.nes.step(action)
