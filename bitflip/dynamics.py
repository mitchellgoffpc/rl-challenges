import torch
import torch.nn.functional as F
from helpers import ReplayBuffer, mean
from bitflip.models import Encoder, Decoder, Dynamics
from bitflip.environment import binary_encode, flip_bits


bit_length = 32
hidden_size = 64
batch_size = 64
report_every = 25

encoder = Encoder(hidden_size, bit_length)
decoder = Decoder(hidden_size, bit_length)
dynamics = Dynamics(hidden_size, bit_length)
optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters(), *dynamics.parameters()], lr=0.0003)

replay_buffer = ReplayBuffer(1024)
decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []


for episode_counter in range(1000):
    episode = []
    old_state = binary_encode(bit_length, torch.randint(0, 2 ** bit_length, ()))
    for step in range(8):
        action = torch.randint(0, bit_length, ())
        new_state = flip_bits(old_state, action)
        episode.append((old_state, action, new_state))
        old_state = new_state
    replay_buffer.push_episode(episode)

    if len(replay_buffer) > batch_size:
        for _ in range(8):
            old_states, actions, new_states = replay_buffer.sample(batch_size)
            old_features = encoder(old_states)
            new_features = encoder(new_states)
            new_feature_preds = dynamics(actions, old_features[:,0])
            old_state_preds = decoder(old_features)
            new_state_preds = decoder(new_features)

            dynamics_loss = F.mse_loss(new_feature_preds, new_features)
            decoder_loss = F.mse_loss(old_state_preds, old_states.float())
            loss = dynamics_loss + decoder_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            dynamics_errors.append(dynamics_loss.item())
            decoder_errors.append(decoder_loss.item())
            decoder_old_accuracies.append((old_state_preds.detach().round() == old_states).float().mean())
            decoder_new_accuracies.append((new_state_preds.detach().round() == new_states).float().mean())

    if episode_counter % report_every == 0:
        print(f"Episode {episode_counter:<4} | "
              f"Dynamics error: {mean(dynamics_errors):.4f} | "
              f"Decoder error: {mean(decoder_errors):.4f} | "
              f"Decoder accuracy: {mean(decoder_old_accuracies)*100:.3f}% | "
              f"Dynamics + Decoder accuracy: {mean(decoder_new_accuracies)*100:.3f}%")
        decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []
