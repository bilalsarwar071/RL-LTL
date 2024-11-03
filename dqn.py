import numpy as np
import random
import gymnasium as gym  # Update gym import

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def __call__(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.asarray(obs_t))
            actions.append(np.asarray(action))
            rewards.append(reward)
            obses_tp1.append(np.asarray(obs_tp1))
            dones.append(done)
        return np.asarray(obses_t), np.asarray(actions), np.asarray(rewards), np.asarray(obses_tp1), np.asarray(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indices = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(indices)


class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(3136, 512)
        self.head = nn.Linear(512, self.n_action)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x.reshape(x.size(0), -1)))
        x = self.head(x)
        return x


class ComposedDQN(nn.Module):
    def __init__(self, dqns, weights=None, or_compose=True):
        super().__init__()
        self.dqns = dqns
        if weights is None:
            self.weights = [1] * len(dqns)
        else:
            self.weights = weights
        self.or_compose = or_compose

    def forward(self, x):
        qs = [self.dqns[i](x) * self.weights[i] for i in range(len(self.weights))]
        q = torch.stack(tuple(qs), 2)
        if self.or_compose:
            return q.max(2)[0]
        return 0.5 * q.sum(2)


def get_value(dqn, obs):
    with torch.no_grad():  # Use torch.no_grad() instead of volatile
        return dqn(obs).max(1)[0].item()

def get_action(dqn, obs):
    with torch.no_grad():  # Use torch.no_grad() instead of volatile
        return dqn(obs).max(1)[1].item()

class Agent(object):
    def __init__(self,
                 env,
                 max_timesteps=200000, #2000000
                 learning_starts=10000,
                 train_freq=4,
                 target_update_freq=1000,
                 learning_rate=1e-4,
                 batch_size=32,
                 replay_buffer_size=1000000,
                 gamma=0.99,
                 eps_initial=1.0,
                 eps_final=0.01,
                 eps_timesteps=500000,
                 print_freq=10):
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Discrete)

        self.env = env
        self.max_timesteps = max_timesteps
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.print_freq = print_freq

        self.eps_schedule = LinearSchedule(eps_timesteps, eps_final, eps_initial)

        self.q_func = DQN(self.env.action_space.n).to(device)
        self.target_q_func = DQN(self.env.action_space.n).to(device)
        self.target_q_func.load_state_dict(self.q_func.state_dict())

        self.optimizer = optim.Adam(self.q_func.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.steps = 0

    def select_action(self, obs):
        sample = random.random()
        eps_threshold = self.eps_schedule(self.steps)
        if sample > eps_threshold:
            obs = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                return self.q_func(obs).max(1)[1].view(1, 1)
        else:
            sample_action = self.env.action_space.sample()
            return torch.tensor([[sample_action]], dtype=torch.long).to(device)

    def train(self):
        obs, _ = self.env.reset()  # gymnasium reset returns additional info
        episode_rewards = [0.0]

        for t in range(self.max_timesteps):
            action = self.select_action(obs)
            new_obs, reward, done, truncated, info = self.env.step(action.item())  # gymnasium step returns more values
            done = done or truncated  # Consider 'truncated' as 'done' for end of episode

            self.replay_buffer.add(obs, action, reward, new_obs, done)
            obs = new_obs

            episode_rewards[-1] += reward
            if done:
                obs, _ = self.env.reset()  # Reset with gymnasium compatibility
                episode_rewards.append(0.0)

            if t > self.learning_starts and t % self.train_freq == 0:
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)
                obs_batch = torch.from_numpy(obs_batch).float().to(device)
                act_batch = torch.from_numpy(act_batch).long().to(device)
                rew_batch = torch.from_numpy(rew_batch).float().to(device)
                next_obs_batch = torch.from_numpy(next_obs_batch).float().to(device)
                not_done_mask = torch.from_numpy(1 - done_mask).float().to(device)

                current_q_values = self.q_func(obs_batch).gather(1, act_batch.squeeze(2)).squeeze()
                next_max_q = self.target_q_func(next_obs_batch).detach().max(1)[0]
                next_q_values = not_done_mask * next_max_q
                target_q_values = rew_batch + (self.gamma * next_q_values)

                loss = F.smooth_l1_loss(current_q_values, target_q_values)

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.q_func.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            if t > self.learning_starts and t % self.target_update_freq == 0:
                self.target_q_func.load_state_dict(self.q_func.state_dict())

            self.steps += 1

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and self.print_freq is not None and len(episode_rewards) % self.print_freq == 0:
                print("--------------------------------------------------------")
                print(f"steps {t}")
                print(f"episodes {num_episodes}")
                print(f"mean 100 episode reward {mean_100ep_reward}")
                print(f"% time spent exploring {int(100 * self.eps_schedule(t))}")
                print("--------------------------------------------------------")

