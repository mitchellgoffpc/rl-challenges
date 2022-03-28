import torch

def flip_bits(state, action):
    next_state = state.clone()
    if state.ndim == 1:
          next_state[action] = not next_state[action]
    else: next_state[torch.arange(len(next_state)), action] = ~next_state[torch.arange(len(next_state)), action]
    return next_state

def binary_encode(bit_length, i):
    return torch.tensor([i >> j & 1 for j in range(bit_length)], dtype=torch.bool)

def build_example(bit_length):
    return binary_encode(bit_length, torch.randint(0, 2 ** bit_length, ()))
