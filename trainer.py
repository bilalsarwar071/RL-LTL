import numpy as np
import torch
import os
from datetime import datetime
from gymnasium.wrappers import RecordVideo  # Updated to gymnasium's RecordVideo wrapper

from dqn import Agent, DQN, ComposedDQN, get_action  # Removed FloatTensor import
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame

# Define the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to determine when to record videos
def video_callable(episode_id):
    return episode_id > 1 and episode_id % 500 == 0

# Training function
def train(path, env):
    # Create a unique folder based on the timestamp to avoid overwriting videos
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    video_path = os.path.join(path, timestamp)
    os.makedirs(video_path, exist_ok=True)
    # Using RecordVideo wrapper to record videos
    env = RecordVideo(env, video_folder=path, episode_trigger=video_callable)
    agent = Agent(env)
    agent.train()
    return agent

# Save agent's model
def save(path, agent):
    torch.save(agent.q_func.state_dict(), path)

# Load the DQN model for evaluation
def load(path, env):
    dqn = DQN(env.action_space.n).to(device)  # Ensure the model is on the correct device
    dqn.load_state_dict(torch.load(path))
    return dqn

# Function to run the environment with a pre-trained DQN
def enjoy(dqn, env, timesteps):
    obs, _ = env.reset()  # Gymnasium reset now returns (obs, info)
    env.render()
    print(env.render())
    for _ in range(timesteps):
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)  # Ensure tensor is on the correct device

        with torch.no_grad():  # Replaced volatile=True with torch.no_grad()
            action = get_action(dqn, obs)

        obs, reward, done, truncated, _ = env.step(action)  # Gymnasium step returns more values (including truncated)
        done = done or truncated  # Handle 'truncated' as 'done'
        env.render()

        if done:
            obs, _ = env.reset()  # Gymnasium reset also returns info
            env.render()

# Compose multiple DQNs with given weights
def compose(dqns, weights):
    return ComposedDQN(dqns, weights)

# Learning function for training a DQN for a specific task
def learn(colour, shape, condition):
    name = colour + shape
    base_path = './models/{}/'.format(name)
    # Initialize CollectEnv with render_mode set to 'rgb_array'
    env = WarpFrame(CollectEnv(goal_condition=condition, render_mode='rgb_array'))  # Ensure the render mode is set correctly
    agent = train(base_path + 'results', env)
    save(base_path + 'model.dqn', agent)

# Main training logic
if __name__ == '__main__':
    learn('purple', 'circle', lambda x: x.colour == 'purple' and x.shape == 'circle')

