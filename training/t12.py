import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment import PrEPDistributionEnv
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Custom callback to log metrics
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.metrics = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Get the latest episode info if available
        if len(self.model.ep_info_buffer) > 0 and hasattr(self.model, "logger"):
            logger = self.model.logger
            metrics = {
                "episode": self.episode_count,
                "reward": self.model.ep_info_buffer[-1]["r"],
                "length": self.model.ep_info_buffer[-1]["l"],
                "time": self.model.ep_info_buffer[-1]["t"],
                "policy_loss": logger.name_to_value.get("train/policy_gradient_loss", np.nan),
                "value_loss": logger.name_to_value.get("train/value_loss", np.nan),
                "entropy_loss": logger.name_to_value.get("train/entropy_loss", np.nan),
                "approx_kl": logger.name_to_value.get("train/approx_kl", np.nan),
                "clip_fraction": logger.name_to_value.get("train/clip_fraction", np.nan),
                "explained_variance": logger.name_to_value.get("train/explained_variance", np.nan)
            }
            self.metrics.append(metrics)
            self.episode_count += 1
        return True

# Training parameters
total_timesteps = 1000000  # 1,000,000 timesteps
seed = 42
models_dir = "models"
logs_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Best PPO hyperparameters
ppo_params = {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "n_steps": 2048,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "batch_size": 64,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "verbose": 1
}

# Create environment
env = DummyVecEnv([lambda: Monitor(PrEPDistributionEnv())])

# Training and evaluation
results = {}
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
config_name = "best2"
print(f"Training {config_name}...")

# Initialize PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    seed=seed,
    **ppo_params
)

# Custom callback for logging
callback = CustomCallback()

# Train model
model.learn(
    total_timesteps=total_timesteps,
    log_interval=10,
    callback=callback,
    progress_bar=True
)
model.save(f"{models_dir}/{config_name}_{timestamp}")

# Evaluate model
mean_reward, std_reward = evaluate_policy(
    model, 
    env, 
    n_eval_episodes=10, 
    deterministic=True
)
results[config_name] = {
    "mean_reward": mean_reward, 
    "std_reward": std_reward
}
print(f"{config_name} Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

# Save training metrics
if callback.metrics:
    metrics_file = f"{logs_dir}/{config_name}_metrics_{timestamp}.csv"
    df_metrics = pd.DataFrame(callback.metrics)
    df_metrics.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")

# Log episode rewards
if hasattr(model, "ep_info_buffer") and model.ep_info_buffer:
    rewards_file = f"{logs_dir}/{config_name}_rewards_{timestamp}.csv"
    df_rewards = pd.DataFrame([
        {
            "episode": i,
            "reward": ep["r"],
            "length": ep["l"],
            "time": ep["t"]
        } 
        for i, ep in enumerate(model.ep_info_buffer)
    ])
    df_rewards.to_csv(rewards_file, index=False)
    print(f"Rewards saved to {rewards_file}")

# Example evaluation rollout
print("\nRunning evaluation rollout...")
eval_env = PrEPDistributionEnv()
obs, _ = eval_env.reset(seed=seed)
for step in range(eval_env.max_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action.item())
    print(f"Step {step}: Action={action.item()}, Reward={reward:.2f}, "
          f"Budget={eval_env.budget:.2f}, Coverage={eval_env.coverage}")
    if done or truncated:
        break
eval_env.close()

# Save results summary
results_file = f"{logs_dir}/results_summary_best_{timestamp}.csv"
pd.DataFrame([{
    "config": config_name,
    "mean_reward": mean_reward,
    "std_reward": std_reward,
    "total_timesteps": total_timesteps,
    "timestamp": timestamp
}]).to_csv(results_file, index=False)
print(f"Results summary saved to {results_file}")

# Cleanup
env.close()
print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Final results:", results)