import numpy as np
from src.env import PRICE_BOUNDS


class BaseAgent:
    """Abstract base class for all pricing agents."""

    def act(self, ctx, t=None):
        raise NotImplementedError

    def update(self, price, ctx, demand):
        raise NotImplementedError


class PolicyAgent(BaseAgent):
    """Wraps a policy function (like the oracle) to satisfy the agent interface."""

    def __init__(self, policy_fn):
        self.policy_fn = policy_fn

    def act(self, ctx, t=None):
        # The policy function is assumed to only need context
        return self.policy_fn(ctx)

    def update(self, price, ctx, demand):
        # The policy (e.g., oracle) does not learn
        pass


class StaticAgent(BaseAgent):
    """Always returns the same fixed price."""

    def __init__(self, fixed_price):
        self.fixed_price = float(fixed_price)

    def act(self, ctx, t=None):
        return self.fixed_price

    def update(self, price, ctx, demand):
        pass


class GreedyOLSAgent(BaseAgent):
    """
    Online ordinary least squares estimator for linear demand model.
    Maintains A = XᵀX + λI, b = Xᵀy, θ̂ = A⁻¹b
    """

    def __init__(self, n_features=6, lambda_reg=1e-3, price_bounds=PRICE_BOUNDS):
        self.A = lambda_reg * np.eye(n_features)
        self.b = np.zeros(n_features)
        self.price_bounds = price_bounds

    def act(self, ctx, t=None):
        try:
            theta_hat = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            theta_hat = np.zeros(self.A.shape[0])

        c_hat = (
            theta_hat[0]
            + theta_hat[1] * ctx[0]
            + theta_hat[2] * ctx[1]
            + theta_hat[3] * ctx[2]
            + theta_hat[4] * ctx[3]
        )
        b_hat = theta_hat[5]

        if b_hat <= 1e-6:
            return 0.5 * (self.price_bounds[0] + self.price_bounds[1])

        price = c_hat / (2.0 * b_hat)
        return float(np.clip(price, *self.price_bounds))

    def update(self, price, ctx, demand):
        phi = np.array([1.0, ctx[0], ctx[1], ctx[2], ctx[3], -price])
        self.A += np.outer(phi, phi)
        self.b += phi * demand


class ThompsonAgent(BaseAgent):
    """
    Bayesian linear regression with Normal prior and Gaussian likelihood.
    Posterior: θ ~ N(μ, Σ)
    """

    def __init__(
        self,
        n_features=6,
        sigma_noise=3.0,
        prior_var=1e3,
        price_bounds=PRICE_BOUNDS,
        seed=None,
    ):
        self.sigma_noise = sigma_noise
        self.price_bounds = price_bounds
        self.rng = np.random.RandomState(seed)
        self.A = (1.0 / prior_var) * np.eye(n_features)
        self.b = np.zeros(n_features)

    def act(self, ctx, t=None):
        try:
            Sigma = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            Sigma = np.eye(self.A.shape[0]) * 1e6

        mu = Sigma.dot(self.b)
        theta_sample = self.rng.multivariate_normal(mu, Sigma)

        c = (
            theta_sample[0]
            + theta_sample[1] * ctx[0]
            + theta_sample[2] * ctx[1]
            + theta_sample[3] * ctx[2]
            + theta_sample[4] * ctx[3]
        )
        b_est = theta_sample[5]

        if b_est <= 1e-6:
            return 0.5 * (self.price_bounds[0] + self.price_bounds[1])

        price = c / (2.0 * b_est)
        return float(np.clip(price, *self.price_bounds))

    def update(self, price, ctx, demand):
        phi = np.array([1.0, ctx[0], ctx[1], ctx[2], ctx[3], -price])
        prec = 1.0 / (self.sigma_noise**2)
        self.A += prec * np.outer(phi, phi)
        self.b += prec * phi * demand
# change 2 — Revise README for Adaptive Pricing Project
