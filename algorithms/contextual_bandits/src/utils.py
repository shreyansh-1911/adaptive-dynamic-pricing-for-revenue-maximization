import numpy as np
import pandas as pd
import os
from scipy import stats

from src.env import (
    DATA_DIR,
    DEFAULT_SEED,
    PRICE_BOUNDS,
    PricingEnv,
    oracle_price_and_expected_revenue,
)


def project_path(*parts):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, *parts)


def generate_run(env: PricingEnv, policy_fn, seed=None, save_path=None):
    """
    Run env with a policy function policy_fn(ctx, t) -> price.
    policy_fn will receive the context and timestep t (0-indexed).
    Forced exploration: for t < env.forced_exploration, policy will be overridden with random price.
    Returns pandas.DataFrame with logs.
    """
    if seed is not None:
        env.rng = np.random.RandomState(int(seed))

    ctx = env.reset()
    rows = []
    done = False
    t = 0
    while not done:
        # forced exploration if in initial window
        if t < env.forced_exploration:
            price = float(env.rng.uniform(env.price_bounds[0], env.price_bounds[1]))
            forced = True
        else:
            price = float(policy_fn(ctx))
            forced = False
        ctx_next, demand, revenue, done, info = env.step(price)

        row = {
            "t": t,
            "day": info["day"],
            "sin_d": ctx[0],
            "cos_d": ctx[1],
            "traffic": ctx[2],
            "competitor_price": info["competitor_price"],
            "price": float(price),
            "demand": float(demand),
            "revenue": float(revenue),
            "oracle_price": float(info["oracle_price"]),
            "oracle_expected_revenue": float(info["oracle_expected_revenue"]),
            "forced_exploration": forced,
        }
        rows.append(row)

        ctx = ctx_next
        t += 1

    df = pd.DataFrame(rows)
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "sim_log.csv")
    df.to_csv(save_path, index=False)
    return df


# ---------------------------
# Example policies
# ---------------------------
def random_policy(ctx, price_bounds=PRICE_BOUNDS, rng=None):
    if rng is None:
        return float(np.random.uniform(price_bounds[0], price_bounds[1]))
    else:
        return float(rng.uniform(price_bounds[0], price_bounds[1]))


def fixed_price_policy(price_value):
    def f(ctx):
        return float(price_value)

    return f


def oracle_policy_factory(theta, pmin, pmax):
    def pol(ctx):
        sin_d, cos_d, traffic, comp_price = ctx
        p_star, _, _, _ = oracle_price_and_expected_revenue(
            theta, sin_d, cos_d, traffic, comp_price, pmin, pmax
        )
        return float(p_star)

    return pol


def find_optimal_static_price(T_STEPS=1000, SEED=DEFAULT_SEED):
    prices_to_test = np.linspace(
        PRICE_BOUNDS[0], PRICE_BOUNDS[1], 25
    )  # Test 25 different prices
    results = []

    print(f"Running static price sweep across {len(prices_to_test)} prices...")

    for price in prices_to_test:
        # 1. Instantiate the environment FOR THIS LOOP.
        #    Crucially, override forced_exploration to 0.
        env = PricingEnv(
            T=T_STEPS,
            seed=SEED,
            forced_exploration=0,  # <--- THIS IS THE FIX
        )
        # print(f"DEBUG: Environment loaded with TRUE_THETA = {env.theta}\n")

        # 2. Create the policy for this price
        policy = fixed_price_policy(price)

        # 3. Run the simulation
        df = generate_run(
            env,
            policy,
            seed=SEED,
            save_path=None,  # We don't need to save 25 intermediate files
        )

        # 4. Get the total revenue
        total_revenue = df["revenue"].sum()
        total_oracle_revenue = df["oracle_expected_revenue"].sum()

        print(
            f"  Price: {price:5.2f}, Total Revenue: {total_revenue:,.2f}, (Oracle Benchmark: {total_oracle_revenue:,.2f})"
        )
        # 5. Store the result
        results.append({"price": price, "total_revenue": total_revenue})

    return results


def get_optimal_price():
    results = find_optimal_static_price()
    # 6. Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # 7. Find the optimal static price
    best_static_run = results_df.loc[results_df["total_revenue"].idxmax()]
    best_price = best_static_run["price"]
    return best_price


def summarize_all(results: dict[str, np.ndarray], baseline: str, oracle: str):
    """
    results: dict of {name: revenue_array}
    baseline: key name of static pricing baseline
    oracle: key name of oracle policy
    """
    # --- Basic summaries ---
    print("\n--- Summary Statistics ---")
    for name, arr in results.items():
        print(f"{name:12s}: mean={arr.mean():.3f}, std={arr.std(ddof=1):.3f}")

    # --- Headroom and capture rates ---
    headroom = (results[oracle].mean() - results[baseline].mean()) / results[
        baseline
    ].mean()
    print(f"\n--- Performance Metrics ---")
    print(f"Theoretical Headroom (Oracle vs {baseline}): {100 * headroom:.2f}%")

    for name, arr in results.items():
        if name in (baseline, oracle):
            continue
        capture_rate = (arr.mean() - results[baseline].mean()) / (
            results[oracle].mean() - results[baseline].mean()
        )
        print(f"{name:12s} Capture Rate: {100 * capture_rate:.2f}%")

    # --- Paired T-Tests vs baseline ---
    print("\n--- Paired T-Tests (vs Baseline) ---")
    for name, arr in results.items():
        if name == baseline:
            continue
        t, p = stats.ttest_rel(arr, results[baseline])
        gain = 100 * (arr.mean() - results[baseline].mean()) / results[baseline].mean()
        print(f"{name:12s}: Gain={gain:.2f}%, t={t:.3f}, p={p:.4f}")

    # --- Head-to-Head comparisons ---
    print("\n--- Head-to-Head Comparisons ---")
    names = [n for n in results if n not in (baseline, oracle)]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            t, p = stats.ttest_rel(results[n1], results[n2])
            gain = 100 * (results[n1].mean() - results[n2].mean()) / results[n2].mean()
            print(f"{n1:12s} vs {n2:12s}: Gain={gain:.2f}%, t={t:.3f}, p={p:.4f}")

    # --- Bootstrap CIs ---
    print("\n--- 95% Confidence Intervals (Bootstrap) ---")
    for name, arr in results.items():
        lo, hi = bootstrap_ci_mean(arr, n_boot=2000)
        print(f"{name:12s}: [{lo:.2f}, {hi:.2f}]")


def bootstrap_ci_mean(data, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi
