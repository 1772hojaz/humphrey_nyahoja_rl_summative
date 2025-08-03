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
from pyvirtualdisplay import Display
import pygame
import imageio
from io import BytesIO
import itertools
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
                metrics["q_loss"] = self.logger.name_to_value.get("train/loss", np.nan)
            elif isinstance(self.model, A2C):
                metrics["policy_loss"] = self.logger.name_to_value.get("train/policy_loss", np.nan)
                metrics["entropy"] = self.logger.name_to_value.get("train/ent_coef_loss", np.nan)
            elif isinstance(self.model, PPO):
                metrics["policy_loss"] = self.logger.name_to_value.get("train/policy_gradient_loss", np.nan)
                metrics["entropy"] = self.logger.name_to_value.get("train/entropy_loss", np.nan)
        self.metrics.append(metrics)
        return True

# Set up virtual display
display = Display(visible=0, size=(1000, 700))
display.start()

# Initialize Pygame
pygame.init()
screen_width = 1000
screen_height = 700
screen = pygame.display.set_mode((screen_width, screen_height))
font = pygame.font.SysFont("arial", 18, bold=True)
small_font = pygame.font.SysFont("arial", 14)
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Load South Africa map
try:
    zaf_map = pygame.image.load("ZAF.png")
    zaf_map = pygame.transform.scale(zaf_map, (600, 700))
except pygame.error as e:
    print(f"Error loading ZAF.png: {e}")
    exit()

# Province polygons
province_polygons = {
    "Gauteng": [(450, 250), (470, 230), (490, 250), (470, 270)],
    "KwaZulu-Natal": [(450, 400), (470, 380), (490, 400), (470, 420)],
    "Western Cape": [(250, 600), (270, 580), (290, 600), (270, 620)],
    "Eastern Cape": [(400, 500), (420, 480), (440, 500), (420, 520)],
    "Limpopo": [(450, 100), (470, 80), (490, 100), (470, 120)],
    "Mpumalanga": [(500, 200), (520, 180), (540, 200), (520, 220)],
    "North West": [(350, 200), (370, 180), (390, 200), (370, 220)],
    "Free State": [(350, 350), (370, 330), (390, 350), (370, 370)],
    "Northern Cape": [(200, 400), (220, 380), (240, 400), (220, 420)]
}

# Metrics saving
def save_metrics(history, regions, output_dir="plots", model_name="Random"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data = {
        "Time Step": history["time_steps"],
        **{f"Coverage_{region['name']}": history["coverage"][i] for i, region in enumerate(regions)},
        **{f"Incidence_{region['name']}": history["incidence"][i] for i, region in enumerate(regions)},
    }
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/{model_name}_metrics_{timestamp}.csv", index=False)
    print(f"Metrics saved to {output_dir}/{model_name}_metrics_{timestamp}.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for i, region in enumerate(regions):
        ax1.plot(history["time_steps"], history["coverage"][i], label=region["name"])
        ax2.plot(history["time_steps"], history["incidence"][i], label=region["name"])
    ax1.set_title(f"Coverage Over Time ({model_name})")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax2.set_title(f"Incidence Over Time ({model_name})")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Incidence")
    ax2.set_ylim(0, 0.01)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_metrics_plot_{timestamp}.png")
    plt.close()
    print(f"Metrics plot saved to {output_dir}/{model_name}_metrics_plot_{timestamp}.png")

# Visualization function
def visualize_episode(model_name="Random", seed=42):
    env = PrEPDistributionEnv(render_dir="plots")
    obs, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0
    action_pulse = {}
    history = {
        "coverage": [[] for _ in range(env.num_regions)],
        "incidence": [[] for _ in range(env.num_regions)],
        "time_steps": [],
        "doses": [],
        "budget": [],
        "funding_dependency": [],
        "actions": []
    }
    step = 0

    pygame.display.set_caption(f"{model_name} PrEP Distribution Simulation")
    region_actions = [3 + i * 4 for i in range(env.num_regions)]

    while step < env.max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                save_metrics(history, env.regions, output_dir="plots", model_name=model_name)
                pygame.quit()
                return total_reward

        if step < len(region_actions):
            action = region_actions[step]
        else:
            action = env.action_space.sample()
        print(f"Step {step}: Action = {action}")
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        for i in range(env.num_regions):
            history["coverage"][i].append(env.coverage[i])
            history["incidence"][i].append(env.incidence[i])
            print(f"Step {step}: {env.regions[i]['name']}: Coverage={env.coverage[i]:.6f}, Incidence={env.incidence[i]:.6f}")
        history["time_steps"].append(env.time_step)
        history["doses"].append(env.doses)
        history["budget"].append(env.budget)
        history["funding_dependency"].append(env.funding_dependency)
        history["actions"].append(env.history["actions"][-1] if env.history["actions"] else "No action")

        action_text = history["actions"][-1]
        print(f"Action Text: {action_text}")
        for i, region in enumerate(env.regions):
            if region["name"] in action_text and ("doses" in action_text or "Clinic" in action_text or "Cold chain" in action_text or "Awareness" in action_text):
                action_pulse[i] = 5

        screen.fill(WHITE)
        screen.blit(zaf_map, (0, 0))

        for i, region in enumerate(env.regions):
            coverage = env.coverage[i]
            incidence = env.incidence[i]
            red = int(255 * (1 - coverage))
            green = int(255 * coverage)
            color = (red, green, 0)
            polygon = province_polygons[region["name"]]
            centroid_x = sum(x for x, _ in polygon) // 4
            centroid_y = sum(y for _, y in polygon) // 4
            if i in action_pulse and action_pulse[i] > 0:
                scale = 1 + 0.05 * np.sin(step * 0.5)
                scaled_polygon = [(centroid_x + (x - centroid_x) * scale, centroid_y + (y - centroid_y) * scale) for x, y in polygon]
                pygame.draw.polygon(screen, YELLOW, scaled_polygon, 3)
                action_pulse[i] -= 1
            else:
                action_pulse.pop(i, None)
            outline_thickness = int(5 * (incidence / 0.01))
            pygame.draw.polygon(screen, BLACK, polygon, outline_thickness)
            pygame.draw.polygon(screen, color, polygon)
            pygame.draw.circle(screen, RED, (centroid_x, centroid_y), 5)
            label = font.render(region["name"], True, BLACK)
            label_rect = label.get_rect(center=(centroid_x, centroid_y - 10))
            screen.blit(label, label_rect)
            inc_label = small_font.render(f"I:{incidence:.6f}", True, BLACK)
            inc_label_rect = inc_label.get_rect(center=(centroid_x, centroid_y + 20))
            screen.blit(inc_label, inc_label_rect)
            cov_label = small_font.render(f"C:{coverage:.6f}", True, BLACK)
            cov_label_rect = cov_label.get_rect(center=(centroid_x, centroid_y + 40))
            screen.blit(cov_label, cov_label_rect)

        pygame.draw.rect(screen, GRAY, (600, 0, 400, 700))
        screen.blit(font.render(f"{model_name} Simulation", True, BLACK), (610, 10))
        screen.blit(font.render(f"Time Step: {env.time_step}/{env.max_steps}", True, BLACK), (610, 40))
        screen.blit(font.render(f"Total Reward: {total_reward:.2f}", True, BLACK), (610, 70))
        screen.blit(font.render(f"Doses: {env.doses:.0f}/{env.max_doses}", True, BLACK), (610, 100))
        screen.blit(font.render(f"Budget: {env.budget:.0f}/{env.max_budget}", True, BLACK), (610, 130))

        doses_ratio = np.clip(env.doses / env.max_doses if env.max_doses != 0 else 0, 0, 1)
        budget_ratio = np.clip(env.budget / env.max_budget if env.max_budget != 0 else 0, 0, 1)
        doses_color = RED if env.doses < 5000 else BLUE
        budget_color = RED if env.budget < 50000 else GREEN
        pygame.draw.rect(screen, doses_color, (610, 150, 200 * doses_ratio, 20))
        pygame.draw.rect(screen, budget_color, (610, 180, 200 * budget_ratio, 20))

        if env.doses < 5000 or env.budget < 50000:
            screen.blit(font.render("WARNING: Low Resources!", True, RED), (610, 210))
        screen.blit(font.render(f"Funding Dependency: {env.funding_dependency:.2f}", True, BLACK), (610, 240))
        action_color = RED if action_text.startswith("Failed") else BLACK
        screen.blit(font.render(f"Action: {action_text[:40]}", True, action_color), (610, 270))

        y_offset = 300
        screen.blit(font.render("Region Details (Gauteng):", True, BLACK), (610, y_offset))
        y_offset += 30
        for key, value in env.regions[0].items():
            if key not in ["pos", "name"]:
                screen.blit(small_font.render(f"{key}: {value:.2f}", True, BLACK), (610, y_offset))
                y_offset += 20

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3))
        for i, region in enumerate(env.regions):
            ax1.plot(history["coverage"][i], label=region["name"][:3])
            ax2.plot(history["incidence"][i])
        ax1.set_title("Coverage Over Time")
        ax2.set_title("Incidence Over Time")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, 1.0)
        ax2.set_ylim(0, 0.01)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_surface = pygame.image.load(buf)
        plot_surface = pygame.transform.scale(plot_surface, (350, 250))
        screen.blit(plot_surface, (610, 400))

        screen.blit(small_font.render("Legend:", True, BLACK), (610, 660))
        pygame.draw.rect(screen, RED, (610, 680, 20, 10))
        screen.blit(small_font.render("Low Coverage", True, BLACK), (635, 680))
        pygame.draw.rect(screen, GREEN, (710, 680, 20, 10))
        screen.blit(small_font.render("High Coverage", True, BLACK), (735, 680))
        screen.blit(small_font.render("Thick Outline: High Incidence", True, BLACK), (610, 700))

        frame = pygame.surfarray.array3d(screen)
        frame = frame.transpose([1, 0, 2])
        frames.append(frame)

        pygame.display.flip()
        clock.tick(5)
        obs = next_obs
        step += 1
        if done:
            break

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gif_path = f"plots/{model_name}_simulation_{timestamp}.gif"
    imageio.mimsave(gif_path, frames, fps=5)
    print(f"GIF saved to {gif_path}")
    save_metrics(history, env.regions, output_dir="plots", model_name=model_name)
    env.close()
    pygame.quit()
    return total_reward

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

    obs, _ = env.reset(seed=seed)
    episode_rewards = []
    log_probs = []
    episode_t = 0

    for t in range(total_timesteps):
        obs_tensor = torch.FloatTensor(obs)
        action_probs = policy(obs_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, done, truncated, info = env.step(action.item())
        episode_rewards.append(reward)
        log_probs.append(log_prob)
        episode_t += 1

        if done or truncated or episode_t >= env.unwrapped.max_steps:
            returns = []
            R = 0
            for r in episode_rewards[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            policy_loss = 0
            for lp, R in zip(log_probs, returns):
                policy_loss += -lp * R
            policy_loss = policy_loss / len(episode_rewards)

            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
            loss = policy_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rewards_log.append(sum(episode_rewards))
            policy_losses.append(policy_loss.item())
            entropies.append(entropy.item())
            episode_count += 1

            obs, _ = env.reset(seed=seed + episode_count)
            episode_rewards = []
            log_probs = []
            episode_t = 0
        else:
            obs = next_obs

        if t % batch_size == 0 and t > 0:
            print(f"Timestep {t}, Episode {episode_count}, Mean Reward: {np.mean(rewards_log[-10:]):.2f}")

    torch.save(policy.state_dict(), f"{models_dir}/{config_name}_{timestamp}.pth")
    log_file = f"{logs_dir}/{config_name}_rewards_{timestamp}.csv"
    df = pd.DataFrame({"r": rewards_log, "l": [env.unwrapped.max_steps] * len(rewards_log), "t": range(len(rewards_log))})
    df.to_csv(log_file, index=False)
    print(f"Rewards logged to {log_file}")

    metrics_file = f"{logs_dir}/{config_name}_metrics_{timestamp}.csv"
    df_metrics = pd.DataFrame({"episode": range(len(policy_losses)), "policy_loss": policy_losses, "entropy": entropies})
    df_metrics.to_csv(metrics_file, index=False)
    print(f"Metrics logged to {metrics_file}")

    policy.eval()
    eval_rewards = []
    for _ in range(10):
        obs, _ = env.reset(seed=seed + _)
        episode_reward = 0
        for _ in range(env.unwrapped.max_steps):
            obs_tensor = torch.FloatTensor(obs)
            action_probs = policy(obs_tensor)
            action = torch.argmax(action_probs).item()
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            if done or truncated:
                break
        eval_rewards.append(episode_reward)
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"{config_name} Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
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

# Hyperparameter grids
dqn_params = {
    "learning_rate": [0.0001, 0.001, 0.01],
    "gamma": [0.95, 0.99],
    "buffer_size": [1000, 10000, 50000],
    "batch_size": [32, 64, 128],
    "exploration_fraction": [0.1, 0.2],
    "target_update_interval": [500, 1000]
}

reinforce_params = {
    "learning_rate": [0.0001, 0.0007, 0.001],
    "gamma": [0.95, 0.99],
    "batch_size": [32, 64],
    "ent_coef": [0.0, 0.01]
}

a2c_params = {
    "learning_rate": [0.0001, 0.0007, 0.001],
    "gamma": [0.95, 0.99],
    "n_steps": [5, 10],
    "ent_coef": [0.0, 0.01]
}

ppo_params = {
    "learning_rate": [0.0001, 0.0003, 0.001],
    "gamma": [0.95, 0.99],
    "n_steps": [1024, 2048],
    "clip_range": [0.1, 0.2]
}

# Training and evaluation
results = {}
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
for model_name, param_grid in [("DQN", dqn_params), ("REINFORCE", reinforce_params), ("A2C", a2c_params), ("PPO", ppo_params)]:
    param_combinations = list(itertools.product(*param_grid.values()))
    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        config_name = f"{model_name}_" + "_".join([f"{k}_{v}" for k, v in param_dict.items()])
        print(f"Training {config_name}...")

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

            env.seed(seed)
            # Then evaluate without seed argument
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)
            results[config_name] = {"mean_reward": mean_reward, "std_reward": std_reward}
            print(f"{config_name} Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

            log_file = f"{logs_dir}/{config_name}_rewards_{timestamp}.csv"
            rewards = model.ep_info_buffer if hasattr(model, "ep_info_buffer") else []
            df = pd.DataFrame(rewards, columns=["r", "l", "t"])
            df.to_csv(log_file, index=False)
            print(f"Rewards logged to {log_file}")

            metrics_file = f"{logs_dir}/{config_name}_metrics_{timestamp}.csv"
            df_metrics = pd.DataFrame(callback.metrics)
            df_metrics.to_csv(metrics_file, index=False)
            print(f"Metrics logged to {metrics_file}")

# Visualize with random actions
print("Visualizing with random actions...")
visualize_episode(model_name="Random", seed=42)

# Plot cumulative rewards
plt.figure(figsize=(12, 8))
for config_name in results.keys():
    log_file = f"{logs_dir}/{config_name}_rewards_{timestamp}.csv"
    df = pd.read_csv(log_file)
    cum_rewards = df["r"].cumsum()
    plt.plot(df.index, cum_rewards, label=config_name)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title(f"Cumulative Rewards Comparison (Trained on {timestamp})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.savefig(f"plots/cumulative_rewards_{timestamp}.png")
plt.close()

# Plot objective function
for model_name in ["DQN", "REINFORCE", "A2C", "PPO"]:
    plt.figure(figsize=(12, 8))
    for config_name in results.keys():
        if model_name not in config_name:
            continue
        metrics_file = f"{logs_dir}/{config_name}_metrics_{timestamp}.csv"
        df = pd.read_csv(metrics_file)
        if model_name == "DQN" and "q_loss" in df.columns:
            plt.plot(df["episode"], df["q_loss"], label=config_name)
        elif model_name in ["REINFORCE", "A2C"] and "policy_loss" in df.columns:
            plt.plot(df["episode"], df["policy_loss"], label=config_name)
        elif model_name == "PPO" and "policy_loss" in df.columns:
            plt.plot(df["episode"], df["policy_loss"], label=config_name)
    plt.xlabel("Episode")
    plt.ylabel("Q-Loss" if model_name == "DQN" else "Policy Loss" if model_name != "PPO" else "Clipped Objective")
    plt.title(f"{'Q-Loss' if model_name == 'DQN' else 'Policy Loss' if model_name != 'PPO' else 'Clipped Objective'} for {model_name} (Trained on {timestamp})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_objective_{timestamp}.png")
    plt.close()

# Plot policy entropy (policy gradient methods only)
for model_name in ["REINFORCE", "A2C", "PPO"]:
    plt.figure(figsize=(12, 8))
    for config_name in results.keys():
        if model_name not in config_name:
            continue
        metrics_file = f"{logs_dir}/{config_name}_metrics_{timestamp}.csv"
        df = pd.read_csv(metrics_file)
        if "entropy" in df.columns:
            plt.plot(df["episode"], df["entropy"], label=config_name)
    plt.xlabel("Episode")
    plt.ylabel("Policy Entropy")
    plt.title(f"Policy Entropy for {model_name} (Trained on {timestamp})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_entropy_{timestamp}.png")
    plt.close()

# Stability analysis
stability = {name: results[name]["std_reward"] for name in results}
print("Stability Analysis (Std Dev of Rewards):")
for name, std in stability.items():
    print(f"{name}: Std Dev = {std:.2f}")

# Save results
results_df = pd.DataFrame({
    "Configuration": list(results.keys()),
    "Mean Reward": [results[name]["mean_reward"] for name in results],
    "Std Reward": [results[name]["std_reward"] for name in results]
})
results_df.to_csv(f"logs/results_{timestamp}.csv", index=False)
print(f"Results saved to logs/results_{timestamp}.csv")

# Cleanup
env.close()
display.stop()
print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CAT")
print("Results:", results)

# Cleanup on exit
import atexit
atexit.register(pygame.quit)