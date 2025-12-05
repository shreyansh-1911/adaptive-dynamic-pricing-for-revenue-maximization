import numpy as np
import pandas as pd
from scipy import stats

from env import (
    DEFAULT_SEED,
    PRICE_BOUNDS,
    PricingEnv,
    oracle_price_and_expected_revenue,
)


# --- Policies ---
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


# --- Runner ---
def generate_run(env: PricingEnv, policy_fn, seed=None):
    if seed is not None:
        env.rng = np.random.RandomState(int(seed))
    ctx = env.reset()
    rows = []
    done = False
    t = 0
    cumulative_rev = 0.0

    while not done:
        # Check forced exploration
        if t < env.forced_exploration:
            price = float(env.rng.uniform(*env.price_bounds))
            forced = True
        else:
            price = float(policy_fn(ctx))
            forced = False

        ctx_next, demand, revenue, done, info = env.step(price)
        cumulative_rev += revenue

        rows.append(
            {
                "t": t,
                "seed": seed,
                "price": price,
                "demand": demand,
                "revenue": revenue,
                "cum_revenue": cumulative_rev,
                "oracle_price": info["oracle_price"],
                "oracle_expected_revenue": info["oracle_expected_revenue"],
                "forced_exploration": forced,
            }
        )
        ctx = ctx_next
        t += 1
    return pd.DataFrame(rows), cumulative_rev


# --- Analysis Helpers ---
def bootstrap_ci_mean(data, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    means = [
        np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)
    ]
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def find_optimal_static_price(T=1000, seed=DEFAULT_SEED):
    prices = np.linspace(PRICE_BOUNDS[0], PRICE_BOUNDS[1], 25)
    results = []
    print(f"Finding Optimal Static Price (sweep={len(prices)})...")

    for p in prices:
        # IMPORTANT: forced_exploration=0 to test pure fixed price
        env = PricingEnv(T=T, seed=seed, forced_exploration=0)
        _, total_rev = generate_run(env, fixed_price_policy(p), seed=seed)
        results.append({"price": p, "total_revenue": total_rev})

    df = pd.DataFrame(results)
    best = df.loc[df["total_revenue"].idxmax()]
    print(
        f"Optimal Static Price: {best['price']:.2f} (Rev: {best['total_revenue']:.2f})"
    )
    return best["price"]


def summarize_all(results: dict, baseline="static", oracle="oracle"):
    print("\n--- Summary Statistics (Final Revenue) ---")
    sorted_names = sorted(results.keys(), key=lambda n: results[n].mean(), reverse=True)
    for name in sorted_names:
        arr = results[name]
        print(f"{name:12s}: mean={arr.mean():,.2f}, std={arr.std(ddof=1):,.2f}")

    base_mean = results[baseline].mean()
    ora_mean = results[oracle].mean()
    headroom = (ora_mean - base_mean) / base_mean
    print(f"\nTheoretical Headroom ({oracle} vs {baseline}): {100 * headroom:.2f}%")

    print("\n--- Comparisons ---")
    for name in sorted_names:
        if name in [baseline, oracle]:
            continue
        arr = results[name]
        cap_rate = (arr.mean() - base_mean) / (ora_mean - base_mean)
        t, p = stats.ttest_rel(arr, results[baseline])
        gain = 100 * (arr.mean() - base_mean) / base_mean
        print(f"{name}: Capture={100 * cap_rate:.1f}%, Gain={gain:.2f}%, p={p:.4f}")
