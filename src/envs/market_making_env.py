# src/envs/market_making_env.py

import gym
from gym import spaces
import numpy as np

class MarketMakingEnv(gym.Env):
    """
    A simplified market making environment.
    The agent controls spread and manages inventory over intraday steps.
    """

    def __init__(self, max_inventory=5, num_steps=1000, seed=None):
        super(MarketMakingEnv, self).__init__()

        self.max_inventory = max_inventory
        self.num_steps = num_steps
        self.current_step = 0
        self.seed(seed)

        # Action space: quote width (0 = tight, 1 = wide)
        self.action_space = spaces.Discrete(3)  # [tight, medium, wide]

        # Observation space: [inventory, midprice, volatility]
        self.observation_space = spaces.Box(
            low=np.array([-max_inventory, 0, 0.0]),
            high=np.array([max_inventory, 100000, 5.0]),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.inventory = 0
        self.midprice = 30000.0  # initial BTC price
        self.volatility = 0.02
        self.cash = 0.0
        self.current_step = 0

        return self._get_obs()

    def _get_obs(self):
        return np.array([self.inventory, self.midprice, self.volatility], dtype=np.float32)

    def step(self, action):
        # Define spread width (tight = smaller spread, wide = larger)
        spread_width = [20, 40, 80][action]
        buy_price = self.midprice - spread_width / 2
        sell_price = self.midprice + spread_width / 2

        # Simulate midprice movement
        shock = np.random.normal(0, self.volatility)
        self.midprice += shock * self.midprice

        # Simulate fills (probabilistic fill based on spread and randomness)
        filled_buy = np.random.rand() < 0.5
        filled_sell = np.random.rand() < 0.5

        reward = 0.0
        if filled_buy and self.inventory < self.max_inventory:
            self.inventory += 1
            self.cash -= buy_price
        if filled_sell and self.inventory > -self.max_inventory:
            self.inventory -= 1
            self.cash += sell_price

        # Mark-to-market PnL
        pnl = self.cash + self.inventory * self.midprice
        reward = pnl - self.last_value
        self.last_value = pnl

        self.current_step += 1
        done = self.current_step >= self.num_steps

        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Inventory: {self.inventory}, Midprice: {self.midprice:.2f}, Cash: {self.cash:.2f}")

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def close(self):
        pass
