import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import matplotlib.pyplot as plt
from agents.r1_agent import train_market_making_agent
from envs.market_making_env import MarketMakingEnv
from stable_baselines3 import DQN

# Train the agent
model = train_market_making_agent(total_timesteps=50000)

# Load trained agent (optional)
# model = DQN.load("models/dqn_market_maker/dqn_market_maker")

# Evaluate the agent
env = MarketMakingEnv()
obs = env.reset()

inventory_history = []
midprice_history = []
pnl_history = []

env.last_value = 0.0  # initialize mark-to-market baseline
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    inventory_history.append(env.inventory)
    midprice_history.append(env.midprice)
    pnl_history.append(env.cash + env.inventory * env.midprice)

# Plot PnL and inventory
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(pnl_history, label="Cumulative PnL")
plt.ylabel("PnL")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(inventory_history, label="Inventory")
plt.ylabel("Inventory")
plt.xlabel("Time Step")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("market_making_results.png", dpi=300)
print("Market making performance plot saved as market_making_results.png")
plt.show()
