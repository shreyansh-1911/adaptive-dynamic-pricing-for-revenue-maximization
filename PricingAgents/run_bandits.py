import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# --- IMPORTS ---
from agents import GreedyOLSAgent, StaticAgent, ThompsonAgent
from agents_dnn import NeuralThompsonAgent
from agents_nonlinear import NonLinearXGBoostBandit
from env import PricingEnv


# --- ADAPTERS ---
class NeuralAdapter:
    def __init__(self, context_dim, price_bounds, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.agent = NeuralThompsonAgent(
            context_dim=context_dim,
            price_bounds=price_bounds,
            hidden_sizes=[64, 64],
            dropout_p=0.10,
            lr=0.01,
        )

    def act(self, ctx, t):
        if t < 50:
            return np.random.uniform(10, 90)
        self.agent._update_stats(ctx)
        return self.agent.select_price(self.agent.normalize(ctx))

    def update(self, price, ctx, demand):
        self.agent.store_transition(self.agent.normalize(ctx), price, price * demand)

    def fit_offline(self, df):
        """Optional offline pretrain hook. Agents that can warm-start may override."""
        # default no-op: override in adapters that support it
        return


class LinearAdapter:
    def __init__(self, agent_cls, **kwargs):
        self.agent = agent_cls(n_features=6, **kwargs)

    def act(self, ctx, t):
        return self.agent.act(np.array([1.0, ctx[0], ctx[1], ctx[2], ctx[3], 0.0]), t)

    def update(self, price, ctx, demand):
        self.agent.update(
            price, np.array([1.0, ctx[0], ctx[1], ctx[2], ctx[3], -price]), demand
        )

    def fit_offline(self, df):
        """Optional offline pretrain hook. Agents that can warm-start may override."""
        # default no-op: override in adapters that support it
        return


class NonLinearAdapter:
    def __init__(self, price_bounds, seed):
        low, high = price_bounds
        # Build a grid consistent with your environment
        self.price_grid = np.linspace(low, high, 50)

        self.agent = NonLinearXGBoostBandit(
            price_grid=self.price_grid,
            K=20,
            sliding_window=2000,
            retrain_every=50,
            forced_exploration=50,
            rng_seed=seed,
        )

    def act(self, ctx, t):
        return self.agent.act(ctx, t)

    def update(self, price, ctx, demand):
        self.agent.update(price, ctx, demand)

    def fit_offline(self, df):
        """Optional offline pretrain hook. Agents that can warm-start may override."""
        # default no-op: override in adapters that support it
        return


class OracleWrapper:
    def act(self, ctx, t):
        return 50.0

    def update(self, p, c, d):
        pass

    def fit_offline(self, df):
        """Optional offline pretrain hook. Agents that can warm-start may override."""
        # default no-op: override in adapters that support it
        return


# --- RUNNER ---
def run_one_agent(name, agent, seed, T):
    env = PricingEnv(T=T, seed=seed)
    env.reset()
    done, t, cumulative = False, 0, 0.0
    logs = []

    while not done:
        ctx = env.get_agent_context() if name == "neural" else env.curr_context
        price = agent.act(ctx, t)
        _, demand, revenue, done, info = env.step(price)

        if name == "oracle":
            revenue, price = info["oracle_expected_revenue"], info["oracle_price"]
        else:
            agent.update(price, ctx, demand)

        cumulative += revenue
        logs.append(
            {
                "seed": seed,
                "t": t,
                "agent": name,
                "price": price,
                "revenue": revenue,
                "cum_revenue": cumulative,
                "oracle_rev": info["oracle_expected_revenue"],
            }
        )
        t += 1
    return logs


def run_experiment(seeds=None, T=4000, results_dir="results"):
    if seeds is None:
        seeds = range(2000, 2005)
    os.makedirs(results_dir, exist_ok=True)
    all_logs = []

    print(f"Starting Experiment (T={T})...")
    for seed in seeds:
        agents = {
            "static": StaticAgent(fixed_price=55.0),
            "linear_thompson": LinearAdapter(ThompsonAgent, seed=seed),
            "greedy_ols": LinearAdapter(GreedyOLSAgent),
            "neural": NeuralAdapter(4, (1.0, 100.0), seed),
            "nonlinear_xgb": NonLinearAdapter((1.0, 100.0), seed),
            "oracle": OracleWrapper(),  # INCLUDED
        }
        for name, agent in agents.items():
            all_logs.extend(run_one_agent(name, agent, seed, T))
        print(f"Seed {seed} done.")

    df = pd.DataFrame(all_logs)
    timestamp = int(time.time())
    df.to_csv(f"{results_dir}/results_{timestamp}.csv", index=False)

    # Stats
    print("\n" + "=" * 50)
    print("FINAL CAPTURE RATES")
    print("=" * 50)
    df_sum = df.groupby(["agent", "seed"])["revenue"].sum().groupby("agent").mean()
    oracle_rev = df_sum["oracle"]
    for agent in df_sum.index:
        print(
            f"{agent:<20} | {df_sum[agent]:,.0f} | {(df_sum[agent] / oracle_rev) * 100:.2f}%"
        )

    # Plotting
    print("Generating plots...")

    # Revenue Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="t", y="cum_revenue", hue="agent", errorbar="sd")
    plt.title("Cumulative Revenue (Includes Oracle & OLS)")
    plt.savefig(f"{results_dir}/rev_{timestamp}.png")

    # Regret Plot
    df["regret"] = df["oracle_rev"] - df["revenue"]
    df["cum_regret"] = df.groupby(["agent", "seed"])["regret"].cumsum()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="t", y="cum_regret", hue="agent", errorbar="sd")
    plt.title("Cumulative Regret (Lower is Better)")
    plt.savefig(f"{results_dir}/regret_{timestamp}.png")


if __name__ == "__main__":
    run_experiment()
