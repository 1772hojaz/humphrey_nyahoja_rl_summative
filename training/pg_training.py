from stable_baselines3 import PPO
from environment.custom_env import PrEPDistributionEnv
from stable_baselines3.common.env_util import make_vec_env

def train_pg():
    env = make_vec_env(PrEPDistributionEnv, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("models/pg/pr_ep_pg_model")
    print("PPO training completed.")

if __name__ == "__main__":
    train_pg()

