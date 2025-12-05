import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from env import (
    DEFAULT_SEED,
    PRICE_BOUNDS,
    PricingEnv,  # Needed to instantiate inside loop for plotting if needed
)
from utils import find_optimal_static_price, fixed_price_policy, generate_run

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. Find the number
    best_price = find_optimal_static_price()

    # 2. Re-run sweep just for plotting (optional, or modify utils to return df)
    # Re-implementing simplified sweep here for plot generation
    prices = np.linspace(PRICE_BOUNDS[0], PRICE_BOUNDS[1], 25)
    results = []
    for p in prices:
        env = PricingEnv(seed=DEFAULT_SEED, forced_exploration=0)
        _, rev = generate_run(env, fixed_price_policy(p), seed=DEFAULT_SEED)
        results.append({"price": p, "total_revenue": rev})

    df = pd.DataFrame(results)

    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["price"], df["total_revenue"], "bo-")
    plt.axvline(
        x=best_price, color="r", linestyle="--", label=f"Optimal: {best_price:.2f}"
    )
    plt.title(f"Static Price Sweep (Seed {DEFAULT_SEED})")
    plt.xlabel("Price")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOTS_DIR}/static_price_revenue_curve.png", dpi=150)
    print("Saved plot.")
