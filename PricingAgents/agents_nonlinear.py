"""
Non-Linear Contextual Bandit — Bootstrapped Ensemble of XGBoost regressors
Rewritten clean version aligned with your runner API: act(), update()
"""

import numpy as np
from xgboost import XGBRegressor


class NonLinearXGBoostBandit:
    def __init__(
        self,
        price_grid,
        K=20,
        sliding_window=2000,
        retrain_every=50,
        forced_exploration=0,
        rng_seed=42,
        base_xgb_params=None,
    ):
        self.price_grid = np.array(price_grid, dtype=float)
        self.K = int(K)
        self.sliding_window = int(sliding_window)
        self.retrain_every = int(retrain_every)
        self.forced_exploration = int(forced_exploration)
        self.rng = np.random.RandomState(rng_seed)

        self.history_contexts = []
        self.history_prices = []
        self.history_demands = []

        self.models = [None] * self.K
        self.model_params = base_xgb_params or {
            "n_estimators": 20,
            "max_depth": 2,
            "learning_rate": 0.1,
            "verbosity": 0,
            "objective": "reg:squarederror",
            "n_jobs": 1,
        }

        self.t = 0
        self._last_retrain_at = 0

    # =================== Runner API ===================
    def act(self, context, t):
        return self._select_price(context)

    def update(self, price, context, demand):
        self._observe(price, demand, context)

    # =================== Core Logic ===================
    def _select_price(self, context):
        context = np.asarray(context, dtype=np.float32).ravel()

        # Exploration phase or untrained ensemble
        if self.t < self.forced_exploration:
            return float(self.rng.choice(self.price_grid))

        # Sample a model for pseudo-Thompson sampling
        k = self.rng.randint(0, self.K)
        model = self.models[k]
        if model is None:
            return float(self.rng.choice(self.price_grid))

        P = self.price_grid.shape[0]

        # Efficient repeated context matrix
        X = np.tile(context, (P, 1)).astype(np.float32)
        # Append prices as last feature
        X = np.hstack([X, self.price_grid.reshape(-1, 1).astype(np.float32)])

        preds = model.predict(X)
        preds = np.maximum(preds, 0.0)

        revenue = preds * self.price_grid

        # Tie-breaking to avoid deterministic bias
        max_rev = revenue.max()
        candidates = np.flatnonzero(np.isclose(revenue, max_rev))
        chosen_idx = int(self.rng.choice(candidates))

        return float(self.price_grid[chosen_idx])

    def _observe(self, price, demand, context):
        self.t += 1

        self.history_contexts.append(np.asarray(context, float).ravel())
        self.history_prices.append(float(price))
        self.history_demands.append(float(demand))

        if len(self.history_demands) > self.sliding_window:
            self.history_contexts = self.history_contexts[-self.sliding_window :]
            self.history_prices = self.history_prices[-self.sliding_window :]
            self.history_demands = self.history_demands[-self.sliding_window :]

        if (self.t - self._last_retrain_at) >= self.retrain_every:
            self._retrain()
            self._last_retrain_at = self.t

    # =================== Training ===================
    def _assemble_Xy(self):
        if len(self.history_demands) == 0:
            return None, None
        Xc = np.vstack(self.history_contexts)
        prices = np.asarray(self.history_prices).reshape(-1, 1)
        X = np.hstack([Xc, prices])
        y = np.asarray(self.history_demands, float)
        return X, y

    def _retrain(self):
        X, y = self._assemble_Xy()
        if X is None or X.shape[0] < 20:
            return
        if y is None:
            return

        n = X.shape[0]
        max_depth_opts = [3, 4, 5]
        lr_opts = [0.01, 0.03, 0.05]

        new_models = []
        for _ in range(self.K):
            idx = self.rng.randint(0, n, size=n)
            Xb, yb = X[idx], y[idx]
            Xb = Xb.astype(np.float32)
            yb = yb.astype(np.float32)

            params = dict(self.model_params)
            params.update(
                {
                    "max_depth": int(self.rng.choice(max_depth_opts)),
                    "learning_rate": float(self.rng.choice(lr_opts)),
                }
            )

            m = XGBRegressor(**params)
            try:
                m.fit(Xb, yb)
            except:
                params.update({"max_depth": 3, "learning_rate": 0.05})
                m = XGBRegressor(**params)
                m.fit(Xb, yb)

            new_models.append(m)

        self.models = new_models
