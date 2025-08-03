import gymnasium as gym
import environment  # This triggers the registration

env = gym.make('PrEPDistribution-v0')

obs, _ = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()  # Random action for testing
    obs, reward, done, truncated, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}\n")

env.close()
