import time
import torch
import torch.nn.functional as F
from helpers import ReplayBuffer, mean
from caves.models import Encoder, Decoder, Dynamics
from caves.environment import CaveEnvironment


max_paths = 4
episode_length = 6
embedding_size = 8
hidden_size = 64
batch_size = 64
report_every = 25

environment = CaveEnvironment(max_paths, embedding_size)
encoder = Encoder(hidden_size, embedding_size)
decoder = Decoder(hidden_size, embedding_size)
dynamics = Dynamics(hidden_size, embedding_size)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(dynamics.parameters()), lr=0.0003)

replay_buffer = ReplayBuffer(1024)
decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []
last_report = time.time()

for episode_counter in range(1000):
    episode = []
    old_state = environment.reset()
    for step in range(episode_length):
        action = torch.randint(0, len(environment.current_room.paths), ())
        new_state = environment.step(action)
        episode.append((old_state, action, new_state))
        old_state = new_state
    replay_buffer.push_episode(episode)

    if len(replay_buffer) > batch_size:
        for _ in range(32):
            old_states, actions, new_states = replay_buffer.sample(batch_size)
            old_features = encoder(old_states)
            new_features = encoder(new_states)
            new_feature_preds = dynamics(actions, old_features[:,0])
            old_state_preds = decoder(old_features)
            new_state_preds = decoder(new_features)

            dynamics_loss = F.mse_loss(new_feature_preds, new_features)
            decoder_loss = F.mse_loss(old_state_preds, old_states.float()) + F.mse_loss(new_state_preds, new_states.float())
            loss = dynamics_loss + decoder_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            dynamics_errors.append(dynamics_loss.item())
            decoder_errors.append(decoder_loss.item())

            # room_embeddings = torch.stack([room.embedding for room in environment.rooms])
            # old_embedding_closest_idxs = torch.linalg.norm(room_embeddings[None] - old_state_preds.view(-1, 1, embedding_size), dim=-1).argmin(dim=-1)
            # old_embedding_idxs = torch.all(room_embeddings[None] == old_states.view(-1, 1, embedding_size), dim=-1).int().argmax(dim=-1)
            # new_embedding_closest_idxs = torch.linalg.norm(room_embeddings[None] - new_state_preds.view(-1, 1, embedding_size), dim=-1).argmin(dim=-1)
            # new_embedding_idxs = torch.all(room_embeddings[None] == new_states.view(-1, 1, embedding_size), dim=-1).int().argmax(dim=-1)
            # decoder_old_accuracies.append((old_embedding_closest_idxs == old_embedding_idxs).float().mean())
            # decoder_new_accuracies.append((new_embedding_closest_idxs == new_embedding_idxs).float().mean())

    if episode_counter % report_every == 0:
        print(f"Episode {episode_counter:<4} | "
              f"Rooms explored: {len(environment.rooms)} | "
              f"Dynamics error: {mean(dynamics_errors):.4f} | "
              f"Decoder error: {mean(decoder_errors):.4f} | "
              f"Decoder accuracy: {mean(decoder_old_accuracies)*100:.3f}% | "
              f"Dynamics + Decoder accuracy: {mean(decoder_new_accuracies)*100:.3f}% | "
              f"Time Taken: {time.time() - last_report:.2f}s")
        decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []
        last_report = time.time()
