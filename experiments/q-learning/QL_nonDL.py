"""
Non Deep Learning approach, using np lookup table for q values
This code is the main entry point for the Q-Learning algorithm implementation. 
It sets up the FrozenLake environment, initializes the Q-table, and runs the training loop to learn the optimal policy.
"""
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Initialize Q-table
Q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Track progress
rewards_per_episode = []
q_values_per_episode = []

# Hyperparameters
alpha = 0.8 # Learning rate
gamma = 0.95 # Discount factor
epsilon = 0.9 # Exploration rate
episodes = 5000

# Training Loop
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    episode_reward = 0
    
    # Decay epsilon over time to reduce exploration
    epsilon = max(0.01, epsilon * 0.999) # er decay
    alpha = max(0.01, alpha * 0.995) # lr decay

    while not done:
        # Choose action (ε-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state, :])
        
        # Take action
        next_state, reward, done, _, _ = env.step(action)
        
        # Modified reward to encourage learning
        if done and reward == 0:  # If fell in hole
            reward = -1

        # Q-table update (Bellman equation)
        Q_table[state, action] += alpha * (reward + gamma * np.max(Q_table[next_state, :]) - Q_table[state, action])
        
        # Append rewards for every step in this episode
        episode_reward += reward

        state = next_state

    # Append total reward and mean Q-values for this episode
    rewards_per_episode.append(episode_reward)
    q_values_per_episode.append(np.mean(Q_table))
    

print("Final Q-Table:")
print(Q_table)

# # Plot the reward curve
# ma_window_size = 200
# smoothed_rewards = np.convolve(rewards_per_episode, np.ones(ma_window_size)/ma_window_size, mode='valid')
# plt.figure(figsize=(10, 6))
# plt.plot(smoothed_rewards)
# plt.title(f'Smoothed Rewards per Episode (Moving Average, Window Size: {ma_window_size})')
# plt.xlabel('Episode')
# plt.ylabel('Average Reward')
# plt.grid(True)
# plt.show()



# Plot both reward and q values in the same graph
ma_window_size = 200
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