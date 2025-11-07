import pandas as pd
import os
import matplotlib.pyplot as plt
from src.env import (
    DEFAULT_SEED,
    DATA_DIR,
)
from src.utils import find_optimal_static_price, project_path

PLOTS_DIR = project_path("plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------
# Quick run when invoked directly
# ---------------------------

if __name__ == "__main__":
    T_STEPS = 1000
    results = find_optimal_static_price(T_STEPS, DEFAULT_SEED)
    # 6. Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # 7. Find the optimal static price
    best_static_run = results_df.loc[results_df["total_revenue"].idxmax()]
    best_price = best_static_run["price"]
    best_revenue = best_static_run["total_revenue"]

    print("\n--- Static Price Analysis Complete ---")
    print(f"Optimal static price: {best_price:.2f}")
    print(f"Revenue at best static price: {best_revenue:,.2f}")

    # 8. Save the required plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df["price"], results_df["total_revenue"], "bo-")
        plt.axvline(
            x=best_price,
            color="r",
            linestyle="--",
            label=f"Optimal Static Price: {best_price:.2f}\nRevenue: {best_revenue:,.2f}",
        )
        plt.title(f"Static Price vs. Total Revenue (T={T_STEPS}, Seed={DEFAULT_SEED})")
        plt.xlabel("Fixed Price ($)")
        plt.ylabel("Total Cumulative Revenue ($)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plot_path = os.path.join(PLOTS_DIR, "static_price_revenue_curve.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved revenue curve plot to {plot_path}")

    except Exception as e:
        print(f"Could not save plot: {e}")

    # 9. Save the summary data
    summary_path = os.path.join(DATA_DIR, "static_price_analysis.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"Saved analysis data to {summary_path}")
# change 8 — Build RL env & revenue computation
# change 33 — Enhance CSV loader
# change 38 — Rewrite setup README
