import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment import PrEPDistributionEnv
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Custom callback to log metrics
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.metrics = []

    def _on_step(self) -> bool:
        metrics = {"episode": len(self.metrics)}
        if hasattr(self.model, "logger"):
            logger = self.model.logger
            if isinstance(self.model, DQN):
                metrics["q_loss"] = logger.name_to_value.get("train/loss", np.nan)
            elif isinstance(self.model, A2C):
                metrics["policy_loss"] = logger.name_to_value.get("train/policy_loss", np.nan)
                metrics["entropy"] = logger.name_to_value.get("train/ent_coef_loss", np.nan)
            elif isinstance(self.model, PPO):
                metrics["policy_loss"] = logger.name_to_value.get("train/policy_gradient_loss", np.nan)
                metrics["entropy"] = logger.name_to_value.get("train/entropy_loss", np.nan)
        self.metrics.append(metrics)
        return True

# REINFORCE Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# REINFORCE Training
def train_reinforce(env, params, total_timesteps, seed, config_name, timestamp):
    torch.manual_seed(seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNetwork(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=params["learning_rate"])
    ent_coef = params["ent_coef"]
    gamma = params["gamma"]
    batch_size = params["batch_size"]

    rewards_log = []
    policy_losses = []
    entropies = []
    episode_count = 0

    # Get the underlying environment for max_steps access
    unwrapped_env = env.envs[0].env if hasattr(env, 'envs') else env.env
    max_steps = getattr(unwrapped_env, 'max_steps', 1000)  # Default to 1000 if not found

    obs = env.reset()
    episode_rewards = []
    log_probs = []
    episode_t = 0

    for t in range(total_timesteps):
        obs_tensor = torch.FloatTensor(obs[0])  # DummyVecEnv returns observations in a list
        action_probs = policy(obs_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, done, info = env.step([action.item()])  # Action needs to be in a list for DummyVecEnv
        episode_rewards.append(reward[0])  # Reward is returned as an array
        log_probs.append(log_prob)
        episode_t += 1

        if done[0] or episode_t >= max_steps:
            # Calculate discounted returns
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Calculate policy loss
            policy_loss = 0
            for lp, R in zip(log_probs, returns):
                policy_loss += -lp * R
            policy_loss = policy_loss / len(log_probs)

            # Calculate entropy bonus
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
            loss = policy_loss - ent_coef * entropy

            # Update policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics
            rewards_log.append(sum(episode_rewards))
            policy_losses.append(policy_loss.item())
            entropies.append(entropy.item())
            episode_count += 1

            # Reset environment
            obs = env.reset()
            episode_rewards = []
            log_probs = []
            episode_t = 0
            
            # Print progress
            if episode_count % 10 == 0:
                avg_reward = np.mean(rewards_log[-10:]) if len(rewards_log) >= 10 else np.mean(rewards_log)
                print(f"Episode {episode_count}, Timestep {t}, Avg Reward: {avg_reward:.2f}")
        else:
            obs = next_obs

    # Save model and logs
    torch.save(policy.state_dict(), f"{models_dir}/{config_name}_{timestamp}.pth")
    
    # Save rewards
    rewards_df = pd.DataFrame({
        "episode": range(len(rewards_log)),
        "reward": rewards_log,
        "length": [max_steps] * len(rewards_log),
        "timestep": np.cumsum([max_steps] * len(rewards_log))
    })
    rewards_df.to_csv(f"{logs_dir}/{config_name}_rewards_{timestamp}.csv", index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        "episode": range(len(policy_losses)),
        "policy_loss": policy_losses,
        "entropy": entropies
    })
    metrics_df.to_csv(f"{logs_dir}/{config_name}_metrics_{timestamp}.csv", index=False)

    # Evaluation
    policy.eval()
    eval_rewards = []
    for i in range(10):
        obs = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            obs_tensor = torch.FloatTensor(obs[0])
            with torch.no_grad():
                action_probs = policy(obs_tensor)
                action = torch.argmax(action_probs).item()
            obs, reward, done, _ = env.step([action])
            episode_reward += reward[0]
            if done[0]:
                break
        eval_rewards.append(episode_reward)
    
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"{config_name} Evaluation - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward, rewards_log, policy_losses, entropies

# Create environment
env = PrEPDistributionEnv(render_dir="plots")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Training parameters
total_timesteps = 50000
seed = 42
models_dir = "models"
logs_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Hyperparameter configurations
dqn_params = [{"learning_rate": 0.0001, "gamma": 0.99, "buffer_size": 10000, "batch_size": 64, "exploration_fraction": 0.1, "target_update_interval": 1000}]
reinforce_params = [{"learning_rate": 0.0007, "gamma": 0.99, "batch_size": 64, "ent_coef": 0.01}]
a2c_params = [{"learning_rate": 0.0007, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01}]
ppo_params = [{"learning_rate": 0.0003, "gamma": 0.99, "n_steps": 1024, "clip_range": 0.2, "ent_coef": 0.02}]

# Training and evaluation
results = {}
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for model_name, param_grid in [("DQN", dqn_params), ("REINFORCE", reinforce_params), ("A2C", a2c_params), ("PPO", ppo_params)]:
    for param_dict in param_grid:
        config_name = f"{model_name}_" + "_".join([f"{k}_{v}" for k, v in param_dict.items()])
        print(f"\nTraining {config_name}...")

        if model_name == "REINFORCE":
            mean_reward, std_reward, rewards_log, policy_losses, entropies = train_reinforce(
                env, param_dict, total_timesteps, seed, config_name, timestamp
            )
            results[config_name] = {"mean_reward": mean_reward, "std_reward": std_reward}
        else:
            callback = CustomCallback()
            if model_name == "DQN":
                model = DQN(policy="MlpPolicy", env=env, verbose=1, seed=seed, exploration_final_eps=0.05, **param_dict)
            elif model_name == "A2C":
                model = A2C(policy="MlpPolicy", env=env, verbose=1, seed=seed, **param_dict)
            else:  # PPO
                model = PPO(policy="MlpPolicy", env=env, verbose=1, seed=seed, **param_dict)

            model.learn(total_timesteps=total_timesteps, log_interval=10, callback=callback)
            model.save(f"{models_dir}/{config_name}_{timestamp}")

            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)
            results[config_name] = {"mean_reward": mean_reward, "std_reward": std_reward}
            print(f"{config_name} Evaluation - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

            # Save rewards and metrics
            if hasattr(model, 'ep_info_buffer'):
                rewards_df = pd.DataFrame(model.ep_info_buffer, columns=["r", "l", "t"])
                rewards_df.to_csv(f"{logs_dir}/{config_name}_rewards_{timestamp}.csv", index=False)
            
            metrics_df = pd.DataFrame(callback.metrics)
            metrics_df.to_csv(f"{logs_dir}/{config_name}_metrics_{timestamp}.csv", index=False)

# Plotting and results saving (same as before)
plt.figure(figsize=(12, 8))
for config_name in results.keys():
    try:
        df = pd.read_csv(f"{logs_dir}/{config_name}_rewards_{timestamp}.csv")
        plt.plot(df.index, df['r'].cumsum(), label=config_name)
    except FileNotFoundError:
        print(f"Warning: Reward file for {config_name} not found")

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title(f"Cumulative Rewards Comparison (Trained on {timestamp})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.savefig(f"plots/cumulative_rewards_{timestamp}.png")
plt.close()

# Save final results
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.reset_index(inplace=True)
results_df.columns = ['Configuration', 'Mean Reward', 'Std Reward']
results_df.to_csv(f"logs/results_{timestamp}.csv", index=False)

env.close()
print("\nTraining completed successfully!")
print("Final Results:")
print(results_df)