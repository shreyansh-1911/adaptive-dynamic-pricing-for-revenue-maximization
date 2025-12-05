"""
DemandModel: fit expected_sales = f(price, context...) on historical CSV rows.
This wrapper is deliberately defensive:
- auto-detects price & sales columns (common names)
- allows user to pass explicit column names
- exposes predict_sales(price, context_row)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

COMMON_PRICE_COLS = ["price", "Price", "p", "P"]
COMMON_SALES_COLS = ["sales", "sales_qty", "demand", "units", "revenue"]


class DemandModel:
    def __init__(self, model_type="gbr", model_kwargs=None, scaler=StandardScaler()):
        self.model_type = model_type
        self.model_kwargs = model_kwargs or {}
        self.scaler = scaler
        self.model = None
        self.fitted = False
        self.feature_cols = None
        self.price_col = None
        self.sales_col = None

    def _guess_cols(self, df, price_col=None, sales_col=None):
        if price_col and price_col in df.columns:
            self.price_col = price_col
        else:
            for c in COMMON_PRICE_COLS:
                if c in df.columns:
                    self.price_col = c
                    break
        if sales_col and sales_col in df.columns:
            self.sales_col = sales_col
        else:
            for c in COMMON_SALES_COLS:
                if c in df.columns:
                    self.sales_col = c
                    break

        if self.price_col is None or self.sales_col is None:
            raise ValueError(
                f"Could not auto-detect price/sales columns. Provide explicit names. "
                f"Found cols: {list(df.columns)}"
            )

    def fit(self, df, price_col=None, sales_col=None, context_cols=None):
        """
        df: pandas DataFrame with historical rows containing price and sales (and optional context)
        price_col, sales_col: strings to override autodetect
        context_cols: list of context columns to use; if None, will use all non-price/non-sales cols
        """
        df = df.reset_index(drop=True).copy()
        self._guess_cols(df, price_col, sales_col)

        if context_cols is None:
            context_cols = [
                c for c in df.columns if c not in [self.price_col, self.sales_col]
            ]

        self.feature_cols = ["_price_"] + context_cols
        X = df[context_cols].copy()
        X["_price_"] = df[self.price_col].astype(float)
        X = X[self.feature_cols].fillna(0.0).astype(float)

        # scale features
        Xs = self.scaler.fit_transform(X)

        y = df[self.sales_col].astype(float).values

        # Choose model
        if self.model_type == "poisson":
            self.model = PoissonRegressor(**self.model_kwargs)
        else:
            # gradient boosting default
            params = dict(n_estimators=200, max_depth=3, learning_rate=0.05)
            params.update(self.model_kwargs)
            self.model = GradientBoostingRegressor(**params)

        self.model.fit(Xs, y)
        self.fitted = True

    def predict_sales(self, price, context_row):
        """
        price: scalar
        context_row: dict-like or pandas Series containing context features (same names used in fit)
        returns expected_sales (float)
        """
        if not self.fitted:
            raise RuntimeError("DemandModel not fitted. Call fit(...) first.")
        # build input vector
        ctx = {}
        for c in self.feature_cols:
            if c == "_price_":
                ctx[c] = price
            else:
                # context_row may be dict or Series
                try:
                    ctx[c] = float(context_row.get(c, 0.0))
                except Exception:
                    ctx[c] = float(0.0)
        X = pd.DataFrame([ctx])[self.feature_cols].astype(float)
        Xs = self.scaler.transform(X)
        yhat = self.model.predict(Xs)[0]
        return max(0.0, float(yhat))
