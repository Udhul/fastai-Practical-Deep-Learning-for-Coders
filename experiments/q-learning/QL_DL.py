import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time

# Create environment
env = gym.make("CartPole-v1")

# Build Deep Q-Network (DQN) using PyTorch
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize model, optimizer and loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = []

# Metrics
rewards_per_episode = []
q_values_per_episode = []

# Training Loop
episodes = 200
initial_start_time = time.time()
# for episode_num, episode in enumerate(range(episodes)):

for episode_num, episode in tqdm(enumerate(range(episodes)), desc='Training Episodes', total=episodes, unit='episodes'):
    start_time = time.time()
    episode_reward = 0

    state = env.reset()[0]
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    done = False
    
    while not done:
        # Choose action (ε-greedy with decay)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = model(state).max(1)[1].item()

        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        # Store experience in memory
        memory.append((state, action, reward, next_state, done))
        if len(memory) > 2000:
            memory.pop(0)
        
        state = next_state
        
        # Train the model using replay
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            states = torch.cat([s[0] for s in minibatch])
            actions = torch.tensor([s[1] for s in minibatch]).to(device)
            rewards = torch.tensor([s[2] for s in minibatch], dtype=torch.float32).to(device)
            next_states = torch.cat([s[3] for s in minibatch])
            dones = torch.tensor([s[4] for s in minibatch], dtype=torch.float32).to(device)
            
            # Compute Q values
            current_q_values = model(states).gather(1, actions.unsqueeze(1))
            next_q_values = model(next_states).max(1)[0].detach()
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))
            
            # Compute loss and update
            loss = criterion(current_q_values.squeeze(), target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    for state, action, reward, _, _ in memory[-episodes:]:
        episode_reward += reward
    rewards_per_episode.append(episode_reward)
    
    # Calculate mean Q-values for this episode
    with torch.no_grad():
        states = torch.cat([m[0] for m in memory[-episodes:]])
        mean_q = model(states).mean().item()
        q_values_per_episode.append(mean_q)

    # Episode Console print
    # episode_time = time.time() - start_time
    # tqdm.write(f'Episode {episode_num + 1} completed in {episode_time:.2f}s. Total elapsed time: {time.time() - initial_start_time:.2f}s')

print("Training complete.")


# Plot results
ma_window_size = 8
smoothed_rewards = np.convolve(rewards_per_episode, np.ones(ma_window_size)/ma_window_size, mode='valid')

plt.figure(figsize=(10, 6))
x_range = range(len(smoothed_rewards))
plt.plot(x_range, smoothed_rewards, label=f'Rewards [MA={ma_window_size}]')
plt.plot(x_range, q_values_per_episode[:len(smoothed_rewards)], label=f'Mean Q-Values')
plt.title('Metrics per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Value')
plt.grid(True)
plt.legend()
plt.show()