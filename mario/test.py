import sys
import time
import pygame
from environment import MarioEnv

def get_action(keys):
  return sum((1 << i) * k for i, k in enumerate(keys))

pygame.init()
screen = pygame.display.set_mode((256, 240)) # (240, 224)

env = MarioEnv()
frame = env.reset()
pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
pygame.display.flip()

running = True
keymap = [pygame.K_j, pygame.K_k, pygame.K_g, pygame.K_h, pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]
keys = [0] * 8
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    elif event.type == pygame.KEYDOWN:
      for i, key in enumerate(keymap):
        if event.key == key:
          keys[i] = 1
      if event.key == pygame.K_r:
        env.reset()
    elif event.type == pygame.KEYUP:
      for i, key in enumerate(keymap):
        if event.key == key:
          keys[i] = 0

  frame, _, _, _ = env.step(get_action(keys))
  pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
  pygame.display.flip()
  time.sleep(0.01)
