import argparse
import os
import time

import numpy as np
from sb3_contrib import RecurrentPPO

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

from utils import get_vectorized_cassie_envs
from config import StudentTrainingConfig


class GRU(nn.Module): # diy GRU to be able to access hidden states of whole sequence
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super().__init__()
        self.layers = nn.ModuleList([nn.GRU(input_size, hidden_size, 1, batch_first=batch_first)])
        for i in range(num_layers-1):
            self.layers.append(nn.GRU(hidden_size, hidden_size, 1, batch_first=batch_first))

    def forward(self, input, init_hidden_states=None):
        full_belief_hidden_state = []
        belief_hidden_state = []
        for i, layer in enumerate(self.layers):
            input, final_hidden = layer(input) if init_hidden_states == None else layer(input, init_hidden_states[i])
            full_belief_hidden_state.append(input)
            belief_hidden_state.append(final_hidden)
        return full_belief_hidden_state, full_belief_hidden_state[-1], belief_hidden_state


class Model(nn.Module):
    def __init__(self, obs_space, action_space, device=None, verbose=0):
        super().__init__()
        if device == None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.verbose = verbose
        self.last_reconstruction = None
        ext_in_dim = obs_space["exteroception_left"].shape[0]
        if "privileged" in obs_space:
            priv_in_dim = obs_space["privileged"].shape[0]

        # exteroceptive encoder
        self.ext_out_dim = 96
        self.exteroceptive_encoder = self._make_encoder(input_dim=ext_in_dim, layer_dims=[256, 160, self.ext_out_dim])

        # belief encoder
        gru_hidden_dim = 192
        gru_num_layers = 2
        gru_in_dim = 2 * self.ext_out_dim + obs_space["proprioception"].shape[0]
        gru_hidden_belief_dim = gru_hidden_dim * gru_num_layers
        self.belief_dim = 2 * self.ext_out_dim

        self.gru = GRU(input_size=gru_in_dim, hidden_size=gru_hidden_dim, num_layers=gru_num_layers, batch_first=True)
        self.attention_encoder = self._make_encoder(input_dim=gru_hidden_dim, layer_dims=[192, 192, 2*self.ext_out_dim])
        self.belief_output_encoder = self._make_encoder(input_dim=gru_hidden_dim, layer_dims=[192, 192, self.belief_dim])

        # belief decoder
        self.encoder_c = self._make_encoder(input_dim=gru_hidden_belief_dim, layer_dims=[512, 512, 2*ext_in_dim])
        self.priv_decoder = self._make_encoder(input_dim=gru_hidden_belief_dim, layer_dims=[384, 128, priv_in_dim]) if "privileged" in obs_space else None
        self.ext_decoder = self._make_encoder(input_dim=gru_hidden_belief_dim, layer_dims=[512, 512, 2*ext_in_dim])

        # lstm
        self.lstm_hidden_size = 256
        lstm_input_dim = obs_space["proprioception"].shape[0] + self.belief_dim
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.lstm_hidden_size, num_layers=2, batch_first=True)
        self.lstm_output_encoder = nn.Linear(self.lstm_hidden_size, action_space.shape[0])

        if self.device == torch.device("cuda"):
            self.cuda()
        if self.verbose:
            print(self)
            total_params = 0
            for p in self.parameters():
                total_params += torch.numel(p)
            print("total params:", total_params)

    def _make_encoder(self, input_dim, layer_dims):
        layers = []
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(input_dim, layer_dims[i]))
            layers.append(nn.LeakyReLU())
            input_dim = layer_dims[i]
        layers.append(nn.Linear(input_dim, layer_dims[-1]))
        return nn.Sequential(*layers)

    @classmethod
    def load(cls, env, load_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(env.observation_space, env.action_space, device)
        model.load_state_dict(torch.load(load_path))
        return model

    @torch.no_grad()
    def predict(self, obs, state, **kwargs):
        if state == None:
            init_belief_state, init_lstm_state = None, None
        else:
            init_belief_state, init_lstm_state = state
        for key, sub_obs in obs.items():
            sub_obs_tensor = torch.tensor(sub_obs, dtype=torch.float, device=self.device)
            if len(sub_obs.shape) == 1: # single timestep
                obs[key] = torch.unsqueeze(sub_obs_tensor, 0)
            elif len(sub_obs.shape) == 2: # batch of sequences of length 1
                obs[key] = torch.unsqueeze(sub_obs_tensor, 1)
            elif len(sub_obs.shape) == 3: # 3D batches
                obs[key] = sub_obs_tensor
            else: # 3D batches
                raise ValueError("Observation space has invalid shape: {}".format(sub_obs.shape))

        ext_latent = self._forward_exteroceptive_encoder(obs["noisy_exteroception_left"], obs["noisy_exteroception_right"])
        belief, full_belief_hidden, belief_hidden = self._forward_belief_encoder(obs["proprioception"], ext_latent, init_belief_state)
        reconstruction = self._forward_belief_decoder(full_belief_hidden, obs["noisy_exteroception_left"], obs["noisy_exteroception_right"])
        action, lstm_state = self._forward_lstm(obs["proprioception"], belief, init_lstm_state)

        self.last_reconstruction = reconstruction
        hidden_states = belief_hidden, lstm_state

        return torch.squeeze(action).cpu().numpy(), hidden_states

    def get_obs_reconstruction(self):
        return {key: tensor.cpu().numpy() for key, tensor in self.last_reconstruction.items()}

    def forward(self, obs):
        ext_latent = self._forward_exteroceptive_encoder(obs["noisy_exteroception_left"], obs["noisy_exteroception_right"])
        belief, full_belief_hidden, _ = self._forward_belief_encoder(obs["proprioception"], ext_latent)
        reconstruction = self._forward_belief_decoder(full_belief_hidden, obs["noisy_exteroception_left"], obs["noisy_exteroception_right"])
        action, _ = self._forward_lstm(obs["proprioception"], belief)
        return action, reconstruction

    def _forward_exteroceptive_encoder(self, noisy_ext_left, noisy_ext_right):
        ext_latent_left = self.exteroceptive_encoder(noisy_ext_left)
        ext_latent_right = self.exteroceptive_encoder(noisy_ext_right)
        return torch.cat([ext_latent_left, ext_latent_right], dim=-1)

    def _forward_belief_encoder(self, proprioception, ext_latent, init_belief_hidden_state=None):
        input = torch.cat((proprioception, ext_latent), dim=-1)
        full_belief_hidden_state, latent_belief, belief_hidden_state = self.gru(input, init_belief_hidden_state) # TODO wrong ordering?

        alpha = torch.sigmoid(self.attention_encoder(latent_belief))
        attention = alpha * ext_latent
        belief = attention + self.belief_output_encoder(latent_belief)

        return belief, full_belief_hidden_state, belief_hidden_state

    def _forward_belief_decoder(self, belief_hidden, noisy_ext_left, noisy_ext_right):
        belief_hidden = torch.cat(belief_hidden, dim=-1)
        alpha = torch.sigmoid(self.encoder_c(belief_hidden))
        attention = alpha * torch.cat([noisy_ext_left, noisy_ext_right], dim=-1)
        ext_reconstruction_cat = torch.chunk(attention + self.ext_decoder(belief_hidden), 2, dim=-1)
        ext_reconstruction = {"left": ext_reconstruction_cat[0], "right": ext_reconstruction_cat[1]}

        if self.priv_decoder:
            priv_reconstruction = self.priv_decoder(belief_hidden)
            ext_reconstruction["privileged"] = priv_reconstruction

        return ext_reconstruction

    def _forward_lstm(self, proprioception, belief, init_lstm_hidden_state=None):
        input = torch.cat((proprioception, belief), -1)
        output, lstm_hidden_state = self.lstm(input, init_lstm_hidden_state)
        action = self.lstm_output_encoder(output)
        return action, lstm_hidden_state


class DataBuffer:
    def __init__(self, n_envs, n_steps, obs_space, action_space, device):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_shapes = {key: subspace.shape for (key, subspace) in obs_space.spaces.items()}
        self.action_shape = action_space.shape
        self.device = device
        self.reset()

    def reset(self):
        self.observations = {}
        for key, shape in self.obs_shapes.items():
            self.observations[key] = np.zeros((self.n_envs, self.n_steps) + shape)
        self.actions = np.zeros((self.n_envs, self.n_steps)+ self.action_shape)
        self.episode_starts = np.zeros((self.n_envs, self.n_steps))
        self.episode_start_indices = None

        self.pos = 0
        self.full = False

    def add(self, obs, actions, episode_starts):
        assert not self.full

        for key, sub_obs in obs.items():
            self.observations[key][:, self.pos] = sub_obs
        self.actions[:, self.pos] = actions
        self.episode_starts[:, self.pos] = episode_starts
        self.pos += 1

        if self.pos == self.n_steps:
            self.full = True
            self._finish()

    def _finish(self):
        for key, sub_obs in self.observations.items():
            self.observations[key] = sub_obs.reshape((self.n_envs * self.n_steps, -1))
        self.actions = self.actions.reshape((self.n_envs * self.n_steps, -1))

        assert self.episode_starts[:, 0].all() == 1
        self.episode_starts = self.episode_starts.flatten()
        self.episode_start_indices = np.where(self.episode_starts == 1)[0]

    def get_batches(self, batch_size):
        random_indices = SubsetRandomSampler(range(len(self.episode_start_indices)-1)) # dropping the last one to make indexing the np arrays much simpler
        batch_sampler = BatchSampler(random_indices, batch_size, drop_last=False)

        for indices in batch_sampler:
            obs_batch = {}
            for key in self.observations:
                sub_obs_sequences = [torch.tensor(self.observations[key][self.episode_start_indices[i]:self.episode_start_indices[i+1]], dtype=torch.float, device=self.device) for i in indices]
                obs_batch[key] = pad_sequence(sub_obs_sequences, batch_first=True)

            action_sequences = [torch.tensor(self.actions[self.episode_start_indices[i]:self.episode_start_indices[i+1]], dtype=torch.float, device=self.device) for i in indices]
            action_batch = pad_sequence(action_sequences, batch_first=True)

            yield action_batch, obs_batch


class Trainer:
    def __init__(self, teacher_path, run_name, train_data_steps=1e5, test_split=0.2, batch_size=32, learning_rate=0.001, save_freq=20):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"using {self.device} device")

        self.teacher_model = RecurrentPPO.load(teacher_path).policy

        self.n_envs = os.cpu_count()
        self.n_steps = max(train_data_steps // self.n_envs, 1)

        config = StudentTrainingConfig()
        self.envs = get_vectorized_cassie_envs(config, self.n_envs)

        self.obs_space = self.envs.observation_space
        self.action_space = self.envs.action_space
        self.teacher_model.observation_space = self.obs_space

        self.train_data_buffer = DataBuffer(self.n_envs, self.n_steps, self.obs_space, self.action_space, self.device)
        self.test_data_buffer = DataBuffer(self.n_envs, int(self.n_steps * test_split), self.obs_space, self.action_space, self.device)

        self.model = Model(self.obs_space, self.action_space, self.device, verbose=1)
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # logging
        self.save_freq = save_freq
        self.run_name = f"{run_name}-{train_data_steps}-{batch_size}-{learning_rate}"
        self.log_dir = os.path.join("student_log", self.run_name)
        self.writer = SummaryWriter(self.log_dir)

    def collect_data(self, data_buffer):
        data_buffer.reset()

        last_lstm_states = None
        last_obs = self.envs.reset()
        last_episode_starts = np.ones((self.n_envs,), dtype=bool)
        while not data_buffer.full:
            with torch.no_grad():
                actions, last_lstm_states = self.teacher_model.predict(last_obs, last_lstm_states, last_episode_starts, deterministic=True)

            new_obs, _, dones, _ = self.envs.step(actions)

            data_buffer.add(last_obs, actions, last_episode_starts)

            last_obs = new_obs
            last_episode_starts = dones

    def train(self, epochs):
        self.collect_data(self.train_data_buffer)
        self.collect_data(self.test_data_buffer)
        print("successfully collected data")
        for epoch in range(epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        self.start_time = time.monotonic()
        running_imitation_loss, running_reconstruction_loss, running_loss = 0, 0, 0

        for i, (actions_t, observations_t) in enumerate(self.train_data_buffer.get_batches(self.batch_size)):
            actions_s, reconstructions = self.model(observations_t)
            imitation_loss, reconstruction_loss, loss = self._get_losses(actions_t, actions_s, observations_t, reconstructions)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_imitation_loss += imitation_loss.item()
            running_reconstruction_loss += reconstruction_loss.item()
            running_loss += loss.item()

        self._log(epoch, running_imitation_loss/i, running_reconstruction_loss/i, running_loss/i)

    def _get_losses(self, actions_t, actions_s, observations_t, reconstructions):
        imitation_loss = self.criterion(actions_s, actions_t)

        reconstruction_label = torch.cat((observations_t["exteroception_left"], observations_t["exteroception_right"]), dim=-1) # TODO privileged
        reconstruction_pred = torch.cat(list(reconstructions.values()), dim=-1)
        reconstruction_loss = self.criterion(reconstruction_pred, reconstruction_label)

        return imitation_loss, reconstruction_loss, imitation_loss + 0.5 * reconstruction_loss

    @torch.no_grad()
    def _eval(self):
        running_imitation_loss, running_reconstruction_loss, running_loss = 0, 0, 0
        for i, (actions_t, observations_t) in enumerate(self.test_data_buffer.get_batches(self.batch_size)):
            actions_s, reconstructions = self.model(observations_t)
            imitation_loss, reconstruction_loss, loss = self._get_losses(actions_t, actions_s, observations_t, reconstructions)

            running_imitation_loss += imitation_loss.item()
            running_reconstruction_loss += reconstruction_loss.item()
            running_loss += loss.item()

        return running_imitation_loss/i, running_reconstruction_loss/i, running_loss/i

    def _log(self, epoch, imitation_loss, reconstruction_loss, loss): # TODO log student model reward?
        self.writer.add_scalar("train/imitation loss", imitation_loss, epoch)
        self.writer.add_scalar("train/reconstruction loss", reconstruction_loss, epoch)
        self.writer.add_scalar("train/total loss", loss, epoch)

        test_imitation_loss, test_reconstruction_loss, test_loss = self._eval()
        self.writer.add_scalar("test/imitation loss", test_imitation_loss, epoch)
        self.writer.add_scalar("test/reconstruction loss", test_reconstruction_loss, epoch)
        self.writer.add_scalar("test/total loss", test_loss, epoch)

        print(f"epoch: {epoch} | test loss: {test_loss:.3f} | train loss: {loss:.3f} | time: {(time.monotonic()-self.start_time):.3f}s")

        if (epoch + 1) % self.save_freq == 0:
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, f"model_{epoch}"))


def get_model_path(teacher_name):
    model_files = os.listdir(os.path.join("teacher_log", teacher_name))
    max_steps = -1
    for filename in model_files:
        if "teacher" in filename:
            steps = int(filename.split("_")[-1].split("k")[0])
            if steps > max_steps:
                max_steps = steps
                teacher_model_name = filename
    return os.path.join("teacher_log", args.teacher_name, teacher_model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('student_name', type=str)
    parser.add_argument('teacher_name', type=str)
    parser.add_argument('train_data_steps', type=int)
    parser.add_argument('epochs', type=int)
    args = parser.parse_args()

    teacher_path = get_model_path(args.teacher_name)
    print("Teacher path:", teacher_path)

    trainer = Trainer(
        teacher_path=teacher_path,
        run_name=args.student_name,
        train_data_steps=args.train_data_steps,
        batch_size=12,
        learning_rate=0.001
    )
    trainer.train(epochs=args.epochs)
