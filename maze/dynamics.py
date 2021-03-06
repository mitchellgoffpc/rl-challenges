import collections
import torch
import torch.nn.functional as F
from helpers import ReplayBuffer, mean, get_contrastive_loss
from maze.models import Encoder, Decoder, Dynamics, NUM_ACTIONS
from maze.environment import MazeEnvironment


grid_w, grid_h = 10, 10
hidden_size = 64
batch_size = 64
report_every = 25

encoder = Encoder(hidden_size, grid_w, grid_h)
decoder = Decoder(hidden_size, grid_w, grid_h)
dynamics = Dynamics(hidden_size)
logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters(), *dynamics.parameters(), logit_scale], lr=0.0003)

env = MazeEnvironment(grid_w, grid_h)
replay_buffer = ReplayBuffer(1024)
decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []

position_counts = collections.defaultdict(lambda: 0)
scores = collections.defaultdict(list)

for episode_counter in range(1000):
    episode = []
    old_state = env.reset()
    old_pos = env.get_info()['position']
    for step in range(10):
        action = torch.randint(0, NUM_ACTIONS, ())
        new_state, _, _, stats = env.step(action)
        episode.append((old_state, action, new_state))

        new_pos = stats['position']
        with torch.no_grad():
            # feats, nfeats = encoder(old_state), encoder(new_state)
            feats = encoder(old_state)
            nfeats = dynamics(action[None,None], feats[None])[0,0]
        scores[position_counts[new_pos]].append(normalize(feats).dot(normalize(nfeats)))

        position_counts[old_pos] += 1
        old_state = new_state
        old_pos = new_pos

    replay_buffer.push_episode(episode)

    if len(replay_buffer) > batch_size:
        for _ in range(10):
            old_states, actions, new_states = replay_buffer.sample(batch_size)
            old_features = encoder(old_states)
            new_features = encoder(new_states)
            new_feature_preds = dynamics(actions, old_features[:,0])
            old_state_preds = decoder(old_features.detach())
            new_state_preds = decoder(new_features.detach())

            bs, ts, _ = old_features.shape
            i = torch.randint(0, ts, (bs,))
            contrastive_loss = get_contrastive_loss(old_features[torch.arange(bs),i], new_features[torch.arange(bs),i], logit_scale)
            dynamics_loss = F.mse_loss(new_feature_preds, new_features)
            decoder_loss = F.mse_loss(old_state_preds, old_states.float())
            loss = dynamics_loss + decoder_loss + contrastive_loss
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
              f"Dynamics + Decoder accuracy: {mean(decoder_new_accuracies)*100:.3f}% | "
              f"Position count for min score: {min(scores.keys(), key=lambda x: mean(scores[x]))} / {max(scores.keys())}")
        decoder_errors, dynamics_errors, decoder_old_accuracies, decoder_new_accuracies = [], [], [], []
        scores.clear()
