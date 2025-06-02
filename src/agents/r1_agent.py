# src/agents/rl_agent.py

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from src.envs.market_making_env import MarketMakingEnv
import os

def train_market_making_agent(total_timesteps=50000, save_path="models/dqn_market_maker"):
    # Wrap the custom env for vectorized training
    env = make_vec_env(lambda: MarketMakingEnv(), n_envs=1)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.001,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        buffer_size=10000,
        learning_starts=1000,
        train_freq=4,
        target_update_interval=100,
        gamma=0.99,
        batch_size=32,
    )

    model.learn(total_timesteps=total_timesteps)

    # Save the model
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/dqn_market_maker")
    print(f"Model saved to {save_path}/dqn_market_maker")

    return model
