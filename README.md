# PrEP Distribution RL

This project implements and evaluates four reinforcement learning algorithms—DQN, REINFORCE, A2C, and PPO—on a custom environment simulating the equitable distribution of Pre-Exposure Prophylaxis (PrEP) across regions in South Africa. The environment includes real-world constraints such as limited budgets, dose supplies, and region-specific demand profiles.

## Key Features

- Custom Gym environment for PrEP distribution
- Multiple RL agents (DQN, REINFORCE, A2C, PPO)
- Constraint-aware action masking
- Automatic logging of rewards, losses, and training stats
- Reward and loss visualization using Matplotlib
- Models, logs, and plots organized and ignored via `.gitignore`
- Git LFS used for tracking large CSV log files
