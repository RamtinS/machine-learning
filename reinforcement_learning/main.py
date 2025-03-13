import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def run(episodes, is_training, render):

  env = gym.make('MountainCar-v0', render_mode="human" if render else None)

  # Divide position and velocity into segments.
  position_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
  velocity_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

  if is_training:
    q_table = np.zeros((len(position_space), len(velocity_space), env.action_space.n)) # 20x20x3 array
  else:
    q_table = np.load('mountain_car.npy')

  learning_rate = 0.9
  discount_factor = 0.9

  epsilon = 1
  epsilon_decay_rate = 2/episodes
  rng = np.random.default_rng()

  rewards_per_episode = np.zeros(episodes)

  for i in range(episodes):
    state = env.reset()[0]  # Starting position and velocity.
    state_p = np.digitize(state[0], position_space)
    state_v = np.digitize(state[1], velocity_space)

    terminated = False  # True when reached goal.

    # The penalty structure is that we get -1 for every action the car takes.
    rewards = 0

    # The process stops when the car has reached the goal or taken more than 1000 actions.
    while not terminated and rewards > -1000:

      if is_training and rng.random() < epsilon:
        action = env.action_space.sample()  # Choose a random action.
      else :
        action = np.argmax(q_table[state_p, state_v, :]) # Choose the best action.

      new_state, reward, terminated, _, _ = env.step(action)
      new_state_p = np.digitize(new_state[0], position_space)
      new_state_v = np.digitize(new_state[1], velocity_space)

      if is_training:
        q_table[state_p, state_v, action] = q_table[state_p, state_v, action] + learning_rate * (
          reward + discount_factor * np.max(q_table[new_state_p, new_state_v, action]) - q_table[state_p, state_v, action]
        )

      state = new_state
      state_p = new_state_p
      state_v = new_state_v

      rewards += reward

    epsilon = max(epsilon - epsilon_decay_rate, 0)

    rewards_per_episode[i] = rewards

  env.close()

  if is_training:
    np.save('mountain_car.npy', q_table)

  mean_rewards = np.zeros(episodes)
  for t in range(episodes):
    mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
  plt.plot(mean_rewards)
  plt.savefig(f'mountain_car.png')

if __name__ == '__main__':
  #run(5000, is_training=True, render=False)

  run(10, is_training=False, render=True)

