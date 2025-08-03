from stable_baselines3 import DQN
from environment.custom_env import PrEPDistributionEnv
from stable_baselines3.common.env_util import make_vec_env

def train_dqn():
    env = make_vec_env(PrEPDistributionEnv, n_envs=1)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("models/dqn/pr_ep_dqn_model")
    print("DQN training completed.")

if __name__ == "__main__":
    train_dqn()
