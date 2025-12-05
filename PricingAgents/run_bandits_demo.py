# run_bandits_demo.py
"""
Demo driver tailored to your sales_data.csv.
Preprocessing maps dataset columns -> numeric context vectors expected by adapters/agents.
Phase-1: train DemandModel on first N warmup rows.
Phase-2: simulate online using demand model for counterfactuals.
"""

import argparse
import os
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from agents import GreedyOLSAgent, StaticAgent, ThompsonAgent
from demand_model import DemandModel

# reuse adapters and agents from your repo
from run_bandits import LinearAdapter, NeuralAdapter, NonLinearAdapter, OracleWrapper
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set_style("darkgrid")
os.makedirs("results", exist_ok=True)


# ---------- Helpers to detect price/sales columns ----------
PRICE_CANDIDATES = ["Unit_Price", "UnitPrice", "price", "Price", "Unit_Price"]
SALES_CANDIDATES = ["Quantity_Sold", "Quantity", "sales", "sales_qty", "demand"]
REVENUE_CANDIDATES = ["Sales_Amount", "Revenue", "sales_amount", "sales_amount"]


def detect_columns(df: pd.DataFrame):
    price_col = None
    sales_col = None
    revenue_col = None
    for c in PRICE_CANDIDATES:
        if c in df.columns:
            price_col = c
            break
    for c in SALES_CANDIDATES:
        if c in df.columns:
            sales_col = c
            break
    for c in REVENUE_CANDIDATES:
        if c in df.columns:
            revenue_col = c
            break

    if price_col is None:
        raise ValueError(f"Could not find price column. Tried: {PRICE_CANDIDATES}")
    # if no explicit sales column but revenue exists, we can compute sales = revenue / price
    if sales_col is None and revenue_col is not None:
        sales_col = "__computed_sales__"
    if sales_col is None:
        raise ValueError(
            f"Could not find sales/quantity column. Tried: {SALES_CANDIDATES} or revenue candidates."
        )
    return price_col, sales_col, revenue_col


# ---------- Preprocessing ----------
def build_preprocessor(df: pd.DataFrame):
    """
    Choose a small set of context features to feed the agents:
    - discount (numeric)
    - Unit_Cost (numeric)
    - Product_Category (categorical -> encoded)
    - Region (categorical -> encoded)
    We encode categories with LabelEncoder and scale continuous features with StandardScaler.
    Returns a function map_row -> (ctx4, ctx6, ctx_dict)
    """
    # candidate feature names in your CSV
    cat_cols_candidates = [
        "Product_Category",
        "Region",
        "Customer_Type",
        "Sales_Channel",
        "Sales_Rep",
    ]
    num_cols_candidates = [
        "Discount",
        "Unit_Cost",
        "Unit_Price",
        "Quantity_Sold",
        "Sales_Amount",
    ]

    # decide which columns exist
    cats = [c for c in cat_cols_candidates if c in df.columns]
    nums = [c for c in num_cols_candidates if c in df.columns]

    # pick two categorical features: Product_Category & Region if present, otherwise first two available
    chosen_cats = []
    for c in ["Product_Category", "Region"]:
        if c in cats:
            chosen_cats.append(c)
    for c in cats:
        if len(chosen_cats) >= 2:
            break
        if c not in chosen_cats:
            chosen_cats.append(c)
    # ensure two categorical placeholders
    while len(chosen_cats) < 2:
        chosen_cats.append(None)

    # pick two numeric features: Discount and Unit_Cost preferred
    chosen_nums = []
    for c in ["Discount", "Unit_Cost"]:
        if c in nums:
            chosen_nums.append(c)
    for c in nums:
        if len(chosen_nums) >= 2:
            break
        if c not in chosen_nums:
            chosen_nums.append(c)
    while len(chosen_nums) < 2:
        chosen_nums.append(None)

    # build encoders/scalers
    encoders = {}
    for c in chosen_cats:
        if c is not None:
            le = LabelEncoder()
            # fillna -> string 'NA' before encoding to avoid errors
            le.fit(df[c].fillna("NA").astype(str))
            encoders[c] = le
        else:
            encoders[c] = None

    scaler = StandardScaler()
    # for numeric scaling, we will scale the two chosen numeric cols and also price later if needed
    num_example = []
    for c in chosen_nums:
        if c is not None:
            num_example.append(df[c].fillna(0.0).astype(float))
        else:
            num_example.append(pd.Series(0.0, index=df.index))
    # build a matrix to fit scaler
    num_matrix = pd.concat(num_example, axis=1).fillna(0.0).astype(float)
    scaler.fit(num_matrix)

    def map_row_to_ctx(row: pd.Series, price_value: float = None):
        """
        Returns:
          - ctx4: np.array shape (4,) used by neural agent fallback: [discount, unit_cost, cat_enc, region_enc]
          - ctx6: np.array shape (6,) used internally by LinearAdapter: [1.0, c0, c1, c2, c3, 0.0] for 'act'
                 (and update will supply -price at last position)
          - ctx_dict: dict of feature name -> value (useful for demand_model)
        """
        # categorical encodings
        cat_vals = []
        for c in chosen_cats:
            if c is None:
                cat_vals.append(0.0)
            else:
                v = row.get(c, "NA")
                try:
                    enc = encoders[c]
                    cat_vals.append(float(enc.transform([str(v)])[0]))
                except Exception:
                    cat_vals.append(0.0)
        # numeric features
        num_vals = []
        for c in chosen_nums:
            if c is None:
                num_vals.append(0.0)
            else:
                num_vals.append(float(row.get(c, 0.0)))
        # scale numeric pair
        scaled = scaler.transform([num_vals])[0]  # length 2

        # ctx4: [scaled_num0, scaled_num1, cat0, cat1]
        ctx4 = np.array([scaled[0], scaled[1], cat_vals[0], cat_vals[1]], dtype=float)

        # ctx6: [1.0, ctx4[0], ctx4[1], ctx4[2], ctx4[3], 0.0]  (last value will be replaced by -price on update)
        ctx6 = np.array([1.0, ctx4[0], ctx4[1], ctx4[2], ctx4[3], 0.0], dtype=float)

        # build ctx_dict for demand model (use a selection of columns)
        ctx_dict = {
            "discount": float(row.get("Discount", 0.0)),
            "unit_cost": float(row.get("Unit_Cost", 0.0)),
            "product_category_enc": cat_vals[0],
            "region_enc": cat_vals[1],
        }
        # include other columns if present (safe)
        for c in [
            "Sales_Rep",
            "Customer_Type",
            "Payment_Method",
            "Sales_Channel",
            "Product_ID",
        ]:
            if c in row.index:
                ctx_dict[c] = row.get(c)

        if price_value is not None:
            ctx_dict["_price_"] = price_value

        return ctx4, ctx6, ctx_dict

    # return mapping func + metadata for later debugging
    meta = {
        "chosen_cats": chosen_cats,
        "chosen_nums": chosen_nums,
        "encoders": encoders,
        "scaler": scaler,
    }
    return map_row_to_ctx, meta


# ---------- Simulated environment wrapper ----------
class SimEnvFromDF:
    def __init__(
        self,
        df: pd.DataFrame,
        demand_model: DemandModel,
        price_col: str,
        sales_col: str,
        mapper,
    ):
        self.df = df.reset_index(drop=True)
        self.dm = demand_model
        self.price_col = price_col
        self.sales_col = sales_col
        self.map_row = mapper

    def get_row(self, t: int):
        return self.df.loc[t]

    def step(self, t: int, price: float):
        row = self.df.loc[t]
        ctx4, ctx6, ctx_dict = self.map_row(row, price_value=price)
        sales = self.dm.predict_sales(price, ctx_dict)
        revenue = float(sales * price)
        return sales, revenue, ctx4, ctx6, ctx_dict


# ---------- Adapter factory (keeps same names as your repo) ----------
def make_adapter(name: str, seed=42, price_bounds=(1.0, 100.0)):
    if name == "static":
        return StaticAgent(fixed_price=55.0)
    if name == "linear_thompson":
        return LinearAdapter(ThompsonAgent, seed=seed)
    if name == "greedy_ols":
        return LinearAdapter(GreedyOLSAgent)
    if name == "nonlinear_xgb":
        return NonLinearAdapter(price_bounds, seed)
    if name == "neural":
        return NeuralAdapter(4, price_bounds, seed)  # ctx4 is shape (4,)
    if name == "oracle":
        return OracleWrapper()
    raise ValueError("unknown adapter")


# ---------- Main demo runner ----------
def run_demo(
    csv_path: str, n_warmup: int = 50, agents_to_run=None, random_seed: int = 42
):
    df = pd.read_csv(csv_path)
    n = len(df)
    if n_warmup >= n:
        raise ValueError("n_warmup must be less than dataset length")

    price_col, sales_col, revenue_col = detect_columns(df)

    # If sales must be computed from revenue/price
    if sales_col == "__computed_sales__":
        # compute from first available revenue column
        rev_col = None
        for c in REVENUE_CANDIDATES:
            if c in df.columns:
                rev_col = c
                break
        if rev_col is None:
            raise RuntimeError("No revenue column to compute sales from.")
        df[sales_col] = df[rev_col] / df[price_col]

    # Build preprocessor mapping function
    mapper, meta = build_preprocessor(df)

    # Phase-1: fit demand model on first n_warmup rows
    dm = DemandModel()
    # demand_model will auto-detect price/sales columns from this slice; ensure column names exist
    # create a slice df_for_fit that contains price & sales & context columns expected (mapper uses column names)
    df_for_fit = df.iloc[:n_warmup].copy()
    # ensure Discount and Unit_Cost exist (fill zeros if missing)
    if "Discount" not in df_for_fit.columns:
        df_for_fit["Discount"] = 0.0
    if "Unit_Cost" not in df_for_fit.columns:
        df_for_fit["Unit_Cost"] = 0.0

    # For demand model fit, add any encoded categorical columns as numeric columns the mapper uses
    # We'll create numeric proxies for chosen categories (so demand_model sees the same encodings)
    map_row, _ = build_preprocessor(df_for_fit)
    enc_rows = []
    for idx, row in df_for_fit.iterrows():
        _, _, ctx_dict = map_row(row)
        enc_rows.append(ctx_dict)

    df_enc = pd.DataFrame(enc_rows)

    # Attach price and sales
    df_enc[price_col] = df_for_fit[price_col].astype(float).values
    df_enc["__sales__"] = df_for_fit[sales_col].astype(float).values

    # --- CRITICAL FIX: keep ONLY numeric columns ---
    df_enc = df_enc.select_dtypes(include=[np.number])

    # Rename for demand model
    df_fit = df_enc.rename(columns={price_col: "price", "__sales__": "sales"})

    # Fit demand model
    dm.fit(df_fit, price_col="price", sales_col="sales")

    # Build simulation env wrapper
    sim = SimEnvFromDF(df, dm, price_col, sales_col, mapper)

    if agents_to_run is None:
        agents_to_run = [
            "static",
            "linear_thompson",
            "greedy_ols",
            "nonlinear_xgb",
            "neural",
            "oracle",
        ]

    results = []
    np.random.seed(random_seed)

    for agent_name in agents_to_run:
        print(f"Running agent {agent_name}...")
        agent = make_adapter(agent_name, seed=random_seed)
        # optional warm-start
        if hasattr(agent, "fit_offline"):
            try:
                agent.fit_offline(df.iloc[:n_warmup])
            except Exception:
                pass

        cumulative = 0.0
        for t in range(n_warmup, n):
            # get the row and contexts
            row = df.loc[t]
            # map to contexts
            ctx4, ctx6, ctx_dict = mapper(row)

            # choose how to call agent.act depending on adapter's expected signature
            # neural adapter expects ctx array of length 4 (we pass ctx4)
            # linear adapter expects ctx array of length 6 (we pass ctx6)
            try:
                if agent_name == "neural":
                    price = agent.act(ctx4, t)
                else:
                    # best-effort: try ctx6 first
                    price = agent.act(ctx6, t)
            except Exception:
                # fallback to ctx4
                try:
                    price = agent.act(ctx4, t)
                except Exception:
                    # random fallback
                    price = float(
                        np.random.uniform(
                            0.8 * row.get(price_col, 50), 1.2 * row.get(price_col, 50)
                        )
                    )

            # simulate step using demand model
            sales, revenue, used_ctx4, used_ctx6, used_ctx_dict = sim.step(t, price)

            # agent update: adapters expect update(price, ctx, demand) in various formats
            try:
                agent.update(price, used_ctx6, sales)
            except Exception:
                try:
                    agent.update(price, used_ctx4, sales)
                except Exception:
                    try:
                        agent.update(price, used_ctx_dict, sales)
                    except Exception:
                        pass

            cumulative += revenue
            results.append(
                {
                    "agent": agent_name,
                    "t": t,
                    "price": price,
                    "sales": sales,
                    "revenue": revenue,
                    "cum_revenue": cumulative,
                }
            )

    df_res = pd.DataFrame(results)
    ts = int(time.time())
    out_csv = f"results/results_demo_{ts}.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Plot cumulative revenue per agent
    plt.figure(figsize=(10, 6))
    df_res["cum_revenue"] = df_res.groupby("agent")["revenue"].cumsum()
    sns.lineplot(data=df_res, x="t", y="cum_revenue", hue="agent")
    plt.title("Cumulative Revenue (demo)")
    plt.savefig(f"results/rev_demo_{ts}.png")
    plt.close()
    print("Plots saved to results/")

    # Save metadata for reproducibility
    meta_out = {"chosen_cats": meta["chosen_cats"], "chosen_nums": meta["chosen_nums"]}
    pd.to_pickle(meta_out, f"results/preproc_meta_{ts}.pkl")
    print("Preprocessing metadata saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/sales_data.csv")
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    run_demo(args.csv, n_warmup=args.warmup)
