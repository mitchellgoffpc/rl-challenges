import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from helpers import ReplayBuffer, mean, get_contrastive_loss
from mario.models import Encoder, Decoder, Dynamics, NUM_ACTIONS
from mario.environment import MarioEnv

hidden_size = 64
batch_size = 64
report_every = 25

device = torch.device('mps')
encoder = Encoder(hidden_size).to(device)
# decoder = Decoder(hidden_size)
# dynamics = Dynamics(hidden_size)
logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07))).to(device)
optimizer = torch.optim.Adam([*encoder.parameters()], lr=0.0003)

# env = MarioEnv()
# replay_buffer = ReplayBuffer(1024)
decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []
contrastive_errors = []
load_times = []
step_times = []

for step in range(10000):
    st = time.monotonic()
    episodes = np.random.randint(0, 900, size=batch_size)
    fidxs = np.random.randint(0, 60, size=(batch_size, 2))
    data = []
    for i, (s, e) in zip(episodes, fidxs):
        ep = np.load(f'mario/episodes/{i:05d}.npz')['arr_0']
        data.append((ep[s], ep[e]))
    data = torch.as_tensor(np.array(data)).permute(0,1,4,2,3).contiguous().to(device)
    load_times.append(time.monotonic() - st)

    # old_states, actions, new_states = replay_buffer.sample(batch_size)
    old_states, new_states = data[:,0], data[:,1]
    old_features = encoder(old_states)
    new_features = encoder(new_states)
    # new_feature_preds = dynamics(actions, old_features[:,0])
    # old_state_preds = decoder(old_features.detach())
    # new_state_preds = decoder(new_features.detach())

    contrastive_loss = get_contrastive_loss(old_features, new_features, logit_scale, device=device)
    # dynamics_loss = F.mse_loss(new_feature_preds, new_features)
    # decoder_loss = F.mse_loss(old_state_preds, old_states.float())
    # loss = dynamics_loss + decoder_loss + contrastive_loss
    loss = contrastive_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    contrastive_errors.append(contrastive_loss.item())
    step_times.append(time.monotonic() - st)

    # dynamics_errors.append(dynamics_loss.item())
    # decoder_errors.append(decoder_loss.item())
    # print(time.time() - st)
    # decoder_old_accuracies.append((old_state_preds.detach().round() == old_states).float().mean())
    # decoder_new_accuracies.append((new_state_preds.detach().round() == new_states).float().mean())

    if step % report_every == 0:
        print(f"Step {step:<4} | "
              f"Contrastive error: {mean(contrastive_errors):.4f} | "
              f"Load times: {mean(load_times):.1f} | "
              f"Step times: {mean(step_times):.1f} | "
              # f"Dynamics error: {mean(dynamics_errors):.4f} | "
              # f"Decoder error: {mean(decoder_errors):.4f} | "
              # f"Decoder accuracy: {mean(decoder_old_accuracies)*100:.3f}% | "
              # f"Dynamics + Decoder accuracy: {mean(decoder_new_accuracies)*100:.3f}%"
        )
        decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []
        contrastive_errors = []
        load_times, step_times = [], []
        torch.save(encoder.state_dict(), f'checkpoint.pt')
