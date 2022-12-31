
import os
import torch
from pathlib import Path
from nes_py import NESEnv as RawNESEnv

class NESEnv:
  A      = 1 << 0
  B      = 1 << 1
  SELECT = 1 << 2
  START  = 1 << 3
  UP     = 1 << 4
  DOWN   = 1 << 5
  LEFT   = 1 << 6
  RIGHT  = 1 << 7
  
  def __init__(self, name):
    self.nes = RawNESEnv(str(Path(__file__).parent / f"roms/{name}.nes"))

  def reset(self):
    return self.nes.reset()

  def wait(self, n_frames):
    for _ in range(n_frames):
      frame, _, _, _ = self.nes.step(0)
    return frame.copy()

  def step(self, action, wait=None):
    if wait is not None:
      self.nes.step(action)
      return self.wait(wait)
    else:
      # ugh who thought it was a good idea to reuse the obs buffer...
      obs, state, data, info = self.nes.step(action)
      return obs.copy(), state, data, info


class MarioEnv(NESEnv):
  def __init__(self):
    super().__init__('mario')

  def reset(self):
    super().reset()
    self.wait(40)
    return self.step(self.START, wait=180)

class ZeldaEnv(NESEnv):
  def __init__(self):
    super().__init__('zelda')

  def reset(self):
    super().reset()
    self.wait(40)
    self.step(self.START, wait=20)
    self.step(self.START, wait=10)
    self.step(self.A, wait=10)
    for _ in range(3):
      self.step(self.SELECT, wait=1)
    self.step(self.START, wait=20)
    return self.step(self.START, wait=120)
