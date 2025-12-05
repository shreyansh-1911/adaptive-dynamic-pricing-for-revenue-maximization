import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.env import PRICE_BOUNDS, TRUE_THETA, PricingEnv, GAUSSIAN_NOISE_SIGMA
from src.agents import PolicyAgent, StaticAgent, GreedyOLSAgent, ThompsonAgent
from src.utils import (
    get_optimal_price,
    summarize_all,
    oracle_policy_factory,
    project_path,
)

OPTIMAL_PRICE = get_optimal_price()


# ---------------------------------------------------------
# Core Experiment
# ---------------------------------------------------------
def run_one(name, agent, seed, T=1000, results_dir=None):
    if results_dir is None:
        results_dir = project_path("results")
    env = PricingEnv(T=T, seed=seed)
    forced_exp = int(env.forced_exploration)

    ctx = env.reset()
    done = False
    t = 0
    logs, cumulative = [], 0.0

    while not done:
        # forced random exploration
        if t < forced_exp:
            price = float(
                env.rng.uniform(
                    low=env.price_bounds[0], high=env.price_bounds[1], size=None
                )
            )
            forced_flag = True
        else:
            price = agent.act(ctx, t)
            forced_flag = False

        ctx_next, demand, revenue, done_flag, info = env.step(price)
        agent.update(price, ctx, demand)
        cumulative += revenue

        logs.append(
            {
                "seed": seed,
                "t": t,
                "agent": name,
                "price": price,
                "demand": demand,
                "revenue": revenue,
                "cum_revenue": cumulative,
                "oracle_price": info["oracle_price"],
                "oracle_rev": info["oracle_expected_revenue"],
                "forced": forced_flag,
            }
        )

        if done_flag:
            done = True
        ctx = ctx_next
        t += 1

    return logs, cumulative


# ---------------------------------------------------------
# Multi-Seed Runner
# ---------------------------------------------------------
def run_experiment(seeds=None, T=1000, results_dir=None, plots_dir=None):
    if results_dir is None:
        results_dir = project_path("results")
    if plots_dir is None:
        plots_dir = project_path("plots")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    if seeds is None:
        seeds = range(2000, 2010)

    all_logs, final_rows = [], []
    start = time.time()
    # Create the oracle policy function
    oracle_policy_fn = oracle_policy_factory(
        TRUE_THETA, PRICE_BOUNDS[0], PRICE_BOUNDS[1]
    )

    for seed in seeds:
        # 1. Instantiate FRESH agents *inside* the seed loop
        agents_to_run = {
            "static": StaticAgent(fixed_price=OPTIMAL_PRICE),
            "greedy_ols": GreedyOLSAgent(),
            "thompson": ThompsonAgent(sigma_noise=GAUSSIAN_NOISE_SIGMA, seed=seed),
            "oracle": PolicyAgent(policy_fn=oracle_policy_fn),
        }

        # This dictionary will hold the final revenue for EACH agent for THIS seed
        seed_final_revenues = {"seed": seed}

        for agent_name, agent_instance in agents_to_run.items():
            # 2. run_one now returns (logs, final_revenue_value)
            logs, final_rev_value = run_one(agent_name, agent_instance, seed, T=T)

            all_logs.extend(logs)  # Use .extend() for lists
            seed_final_revenues[agent_name] = final_rev_value

        # 3. Append the *single, complete* row for this seed
        final_rows.append(seed_final_revenues)
        print(f"Seed {seed} done: {seed_final_revenues}")

    print("Runtime:", round(time.time() - start, 2), "s")

    # 4. Create DataFrames. df_final now has the correct structure.
    df = pd.DataFrame(all_logs)
    df_final = pd.DataFrame(final_rows)

    df.to_csv(f"{results_dir}/results.csv", index=False)
    df_final.to_csv(f"{results_dir}/final_summary.csv", index=False)

    # Plot cumulative revenue
    grp = df.groupby(["agent", "t"])["cum_revenue"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(9, 6))
    for a in grp.agent.unique():
        g = grp[grp.agent == a]
        plt.plot(g.t, g["mean"], label=a)
        plt.fill_between(g.t, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative revenue")
    plt.title("Mean ± SD cumulative revenue across seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/cumulative_revenue.png", dpi=150)

    # Regret
    oracle = df.groupby(["seed", "t"])["oracle_rev"].mean().reset_index()
    rev = df.groupby(["agent", "seed", "t"])["revenue"].sum().reset_index()
    merged = rev.merge(oracle, on=["seed", "t"])
    merged["regret"] = merged["oracle_rev"] - merged["revenue"]
    merged["cum_regret"] = merged.groupby(["agent", "seed"])["regret"].cumsum()
    reg_grp = (
        merged.groupby(["agent", "t"])["cum_regret"].agg(["mean", "std"]).reset_index()
    )

    plt.figure(figsize=(9, 6))
    for a in reg_grp.agent.unique():
        g = reg_grp[reg_grp.agent == a]
        plt.plot(g.t, g["mean"], label=a)
        plt.fill_between(g.t, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative regret")
    plt.title("Mean ± SD cumulative regret across seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/cumulative_regret.png", dpi=150)

    # Plot price trajectories
    plt.figure(figsize=(9, 6))
    # Filter to only learning agents
    df_prices = df[~df["agent"].isin(["static", "oracle"])]
    sns.lineplot(
        data=df_prices, x="t", y="price", hue="agent", errorbar="sd", estimator="mean"
    )
    # Plot the true optimal static price as a reference
    plt.axhline(
        y=OPTIMAL_PRICE,
        color="r",
        linestyle="--",
        label=f"Optimal Static ({OPTIMAL_PRICE:.2f})",
    )
    plt.title("Mean Price Trajectories (Learning Agents)")
    plt.xlabel("Timestep")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/price_trajectories.png", dpi=150)

    # Statistical evaluation
    th, st, gr, ora = (
        df_final["thompson"],
        df_final["static"],
        df_final["greedy_ols"],
        df_final["oracle"],
    )

    results = {
        "Thompson": th,
        "Greedy OLS": gr,
        "Static": st,
        "Oracle": ora,
    }
    summarize_all(results, baseline="Static", oracle="Oracle")

    return df_final


if __name__ == "__main__":
    run_experiment()
