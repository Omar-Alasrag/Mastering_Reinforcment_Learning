import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym.wrappers import RecordVideo

# Replay Buffer stores past experiences for training the model

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim):

        self.buffer_size = buffer_size
        self.state_dim = state_dim

        self.current_index = 0
        self.current_size = 0

        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros(buffer_size)
        self.rewards = np.zeros(buffer_size)
        self.next_states = np.zeros((buffer_size, state_dim))
        self.dones = np.zeros(buffer_size)
        
 
    def add(self, state, action, reward, next_state, done):

        self.states[self.current_index] = state
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.next_states[self.current_index] = next_state
        self.dones[self.current_index] = done

        self.current_index += 1

        if self.current_index >= self.buffer_size:  
            self.current_index = 0

        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        
        return (
            torch.FloatTensor(self.states[indices]),        # Convert states to float32
            torch.LongTensor(self.actions[indices]),        # Convert actions to int64
            torch.FloatTensor(self.rewards[indices]),       # Convert rewards to float32
            torch.FloatTensor(self.next_states[indices]),   # Convert next_states to float32
            torch.FloatTensor(self.dones[indices])          # Convert dones to float32
        )


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 127)
        self.fc3 = nn.Linear(127, action_dim)

        self.relu = nn.ReLU()

    def forward(self, state):

        x = self.fc1(state)
        x = self.relu(x)     
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def optimize_model(replay_buffer, online_model, target_model, optimizer, batch_size, gamma):
    
    if replay_buffer.current_size < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    q_values = online_model(states)
    selected_q_values = q_values.gather(1, actions.unsqueeze(1))

    with torch.no_grad():
        next_q_values = target_model(next_states)
        max_next_q_values = next_q_values.max(1)[0]
        targets = rewards + gamma * max_next_q_values * (1 - dones)

    loss = nn.MSELoss()(selected_q_values, targets.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()






# Create a unique video folder to avoid overwriting

video_folder = f"videos_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
os.makedirs(video_folder, exist_ok=True)




# Initialize environment with RecordVideo

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: e % 50 == 0)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

learning_rate = 0.001
gamma = 0.99
replay_buffer_size = 50000
batch_size = 64
episodes = 300
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.98


replay_buffer = ReplayBuffer(replay_buffer_size, state_dim)

online_model = QNetwork(state_dim, action_dim)
target_model = QNetwork(state_dim, action_dim)

target_model.load_state_dict(online_model.state_dict())

optimizer = optim.Adam(online_model.parameters(), lr=learning_rate)



for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = online_model(state_tensor)
            action = torch.argmax(q_values).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state       
        total_reward += reward
        optimize_model(replay_buffer, online_model, target_model, optimizer, batch_size, gamma)
        decayed_epsilon = epsilon * epsilon_decay
        epsilon = max(epsilon_min, decayed_epsilon)

    target_model.load_state_dict(online_model.state_dict())
    print(f"Episode {episode}: Total Reward = {total_reward}")

env.close()
