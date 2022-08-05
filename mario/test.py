import time
import pygame
from nes import NES

pygame.init()
screen = pygame.display.set_mode((240, 224))

nes = NES("mario.nes", headless=True, verbose=False, sync_mode=0)
frame = nes.run_frame_headless(run_frames=60)
pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
pygame.display.flip()
time.sleep(1)

frame = nes.run_frame_headless(run_frames=180, controller1_state=[0,0,0,1,0,0,0,0])
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
    elif event.type == pygame.KEYUP:
      for i, key in enumerate(keymap):
        if event.key == key:
          keys[i] = 0

  frame = nes.run_frame_headless(run_frames=1, controller1_state=keys)
  pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
  pygame.display.flip()
  time.sleep(0.01)
