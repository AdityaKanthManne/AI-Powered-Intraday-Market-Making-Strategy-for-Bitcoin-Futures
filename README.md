# AI-Powered Intraday Market Making Strategy for Bitcoin Futures

This project implements an intraday market making engine for Bitcoin perpetual futures, trained using reinforcement learning. It simulates a limit order book (LOB) environment and teaches an agent to optimize bid/ask quoting strategies based on inventory risk, spread, and market volatility.

---

## Objectives

- Simulate a synthetic limit order book environment using free market data
- Use Reinforcement Learning (RL) to train an agent that balances PnL and inventory risk
- Track spread capture, order flow imbalance, and inventory constraints
- Optimize quoting policy for maximizing Sharpe ratio and minimizing adverse selection

---

## Key Components

| Component        | Description                                                      |
|------------------|------------------------------------------------------------------|
| Order Book Model | Simulated market depth and midprice spread evolution             |
| RL Agent         | Deep Q-Learning or Policy Gradient approach                      |
| State Features   | Inventory level, midprice, volatility, order flow imbalance      |
| Reward Function  | Spread earned - inventory penalty - execution risk               |

---

## Tech Stack

- Python (NumPy, pandas, matplotlib)
- OpenAI Gym (custom environment)
- Stable-Baselines3 (free RL library)
- Historical data from Binance (via ccxt or CSV)
- No paid APIs required

---

## Run Instructions

```bash
git clone https://github.com/AdityaKanthManne/AI-Powered-Intraday-Market-Making-Strategy-for-Bitcoin-Futures.git
cd AI-Powered-Intraday-Market-Making-Strategy-for-Bitcoin-Futures
pip install -r requirements.txt
python main.py
```

---

## Output

- Trained RL agent with stable policy
- Midprice chart, quote decisions, and inventory evolution
- Cumulative PnL, spread efficiency, and Sharpe ratio
- Optionally export trades and states to CSV

---

## License

MIT License © 2025 Aditya Kanth Manne  
https://github.com/AdityaKanthManne

---

## Contributions

Open to extensions such as multi-agent simulation, real Binance WebSocket integration, and hybrid supervised + RL models.
