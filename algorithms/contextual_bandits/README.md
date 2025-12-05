# Contextual Bandits for Dynamic Pricing Simulation

This project models a **dynamic pricing problem** as a **Contextual Bandit**.
It implements and compares multiple pricing strategies — from a fixed baseline to an adaptive **Thompson Sampling** agent — to determine the optimal policy for **maximizing cumulative revenue** in a simulated, noisy market.

The experiment demonstrates that adaptive, contextual agents (Thompson Sampling, Greedy OLS) capture **70–75% of the theoretically available revenue gain**, significantly and robustly outperforming the best fixed-price strategy.

---

## 📂 Project Structure

```

dynamic_pricing_bandits/
├── data/                      # Intermediate CSVs and simulation logs
│   └── sim_log.csv
├── plots/                     # Figures from static and bandit experiments
│   ├── cumulative_regret.png
│   ├── cumulative_revenue.png
│   ├── price_trajectories.png
│   └── static_price_revenue_curve.png
├── results/                   # Aggregated experiment outputs
│   ├── final_summary.csv
│   └── results.csv
├── src/                       # Source code (modularized)
│   ├── agents.py              # Agent classes (Static, Greedy OLS, Thompson, Oracle)
│   ├── env.py                 # Pricing environment (contextual bandit)
│   ├── run_bandits.py         # Full multi-seed contextual bandit experiment
│   ├── run_static.py          # Static price optimization baseline
│   ├── utils.py               # Utilities (paths, math, stats)
│   └── **init**.py
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── venv/                      # (Optional) local virtual environment
```

---

## The Environment: A Contextual Bandit Problem

Unlike a simple multi-armed bandit, the reward (revenue) for any given price is **non-stationary** — it depends on a **context vector** that changes at each timestep.

---

### 1. Demand Model (`src/env.py`)

The environment’s “physics” are built on a **linear demand model**.
The expected demand μ for a given price $p$ is:

$$
mu_t = c_t - b \cdot p_t
$$

The observed demand $( d_t )$ includes Gaussian noise:

$$
d_t = \max(0, mu_t + \mathcal{N}(0, \sigma^2))
$$

The agent’s objective is to learn the parameters of this model to maximize total revenue.

---

### 2. Context Vector $(c_t)$

The intercept term $c_t$ is **context-dependent**, varying dynamically with external features:

$$
c_t = \theta_0 + \theta_1 \sin(\text{day}_t) + \theta_2 \cos(\text{day}_t) + \theta_3\, \text{traffic}_t + \theta_4\, \text{comp\_price}_t
$$

Thus, the agent must learn the **6-dimensional parameter vector**:

$$
\theta = [\theta_0, \theta_1, \theta_2, \theta_3, \theta_4, b]
$$

---

### 3. The Action–Reward Loop

| Component       | Description                                               |
| --------------- | --------------------------------------------------------- |
| **State (Sₜ)**  | Context vector: [sin(day), cos(day), traffic, comp_price] |
| **Action (Aₜ)** | Price $( p_t \in [5.00, 50.00] )$                         |
| **Reward (Rₜ)** | Observed revenue $( R_t = p_t \times d_t )$               |
| **Goal**        | Maximize cumulative revenue $( \sum_t R_t )$              |

---

## Agents and Strategies

All agents are implemented in **`src/agents.py`**.

| Agent                 | Description                                                                           | Behavior     |
| --------------------- | ------------------------------------------------------------------------------------- | ------------ |
| **Static (Baseline)** | Plays one fixed, optimal price (computed via `run_static.py`)                         | Non-adaptive |
| **Greedy OLS**        | Performs online linear regression (OLS). Always exploits current parameter estimate.  | Exploitative |
| **Thompson Sampling** | Bayesian posterior sampling over parameters. Trades off exploration and exploitation. | Adaptive     |
| **Oracle**            | Has access to true environment parameters (TRUE_THETA). Serves as the upper bound.    | Optimal      |

---

## Running the Simulation

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Run the Full Multi-Seed Bandit Experiment

```bash
python -m src.run_bandits
```

Results will be saved in the `results/` directory.

---

### 3. (Optional) Run Static Price Analysis (Baseline)

```bash
python -m src.run_static
```

Plots and logs will appear in the `plots/` and `data/` directories respectively.

---

## Results Overview

The experiment was run for **10 independent seeds** with **T = 1000 timesteps**.

| Agent                | Final Revenue | Std. Dev. |
| -------------------- | ------------- | --------- |
| **Oracle**           | 286,503.53    | 1,276.60  |
| **Thompson**         | 281,794.56    | 4,353.42  |
| **Greedy OLS**       | 280,931.49    | 5,109.66  |
| **Static (Optimal)** | 267,110.44    | 1,353.08  |

---

### Performance Metrics

| Metric                                  | Value      |
| --------------------------------------- | ---------- |
| Theoretical Headroom (Oracle vs Static) | **7.26%**  |
| Thompson Capture Rate                   | **75.72%** |
| Greedy OLS Capture Rate                 | **71.27%** |

---

### Statistical Tests

| Comparison                 | Gain (%) | t-Statistic | p-Value |
| -------------------------- | -------- | ----------- | ------- |
| **Thompson vs Static**     | +5.50    | 11.644      | 0.0000  |
| **Greedy OLS vs Static**   | +5.17    | 8.921       | 0.0000  |
| **Oracle vs Static**       | +7.26    | 119.284     | 0.0000  |
| **Thompson vs Greedy OLS** | +0.31    | 1.942       | 0.0840  |

---

## Generated Artifacts

| Directory    | File                             | Description                                |
| ------------ | -------------------------------- | ------------------------------------------ |
| **plots/**   | `cumulative_revenue.png`         | Mean cumulative revenue across agents      |
|              | `cumulative_regret.png`          | Mean regret vs Oracle                      |
|              | `price_trajectories.png`         | Price evolution over time                  |
|              | `static_price_revenue_curve.png` | Baseline static analysis                   |
| **results/** | `final_summary.csv`              | Per-seed total revenue                     |
|              | `results.csv`                    | Full time-step logs                        |
| **data/**    | `sim_log.csv`                    | Raw simulation output from `run_static.py` |

---

## Key Insights

- **Dynamic Pricing is Effective** — The environment exhibits a **7.26% revenue headroom** only capturable through adaptive pricing.
- **Learning Works** — Both **Thompson Sampling** and **Greedy OLS** significantly outperform static pricing (**p < 0.0001**).
- **Thompson ≈ Greedy** — The difference is not statistically significant (**p = 0.084**), implying both converge comparably under noise.

---

## Summary

This simulation validates how **contextual bandits** can solve **realistic dynamic pricing** problems — where demand depends on time-varying and contextual factors.
Results align with research literature: **Bayesian and linear contextual agents** recover a large fraction of the optimal policy’s revenue **without access to true parameters**.
