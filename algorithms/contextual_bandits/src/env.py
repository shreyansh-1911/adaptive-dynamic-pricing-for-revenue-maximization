import numpy as np
import os
from math import pi

# ---------------------------
# Configuration / Defaults
# ---------------------------
DEFAULT_SEED = 42
T_DEFAULT = 1000  # timesteps per episode
PRICE_BOUNDS = (5.0, 50.0)  # [min, max]
FORCED_EXPLORATION = 10
GAUSSIAN_NOISE_SIGMA = 3.0  # observation noise sigma
TRAFFIC_MEAN = 500  # average traffic
TRAFFIC_POISSON = True  # whether traffic is drawn from Poisson (realistic)
COMP_BASE = 25.0
COMP_AMPLITUDE = 15.0
COMP_PERIOD = 30.0
COMP_NOISE_SIGMA = 5.0

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------
# True (hidden) parameters
# theta = [theta0, theta_sin, theta_cos, theta_traffic, theta_comp, b]
# ---------------------------
TRUE_THETA = np.array(
    [
        10.0,  # intercept
        2.0,  # day_sin
        -1.0,  # day_cos
        0.02,  # traffic
        0.5,  # competitor price effect
        1.0,  # b, the price coefficient (positive)
    ],
    dtype=float,
)


# ---------------------------
# Helper math
# ---------------------------
def cyclical_day_features(day_index):
    """day_index: 0..6 -> returns (sin, cos)"""
    angle = 2.0 * pi * (day_index % 7) / 7.0
    return float(np.sin(angle)), float(np.cos(angle))


def compute_c_t(theta, day_sin, day_cos, traffic, comp_price):
    """Compute c_t (demand intercept part) before subtracting b*p"""
    return (
        theta[0]
        + theta[1] * day_sin
        + theta[2] * day_cos
        + theta[3] * traffic
        + theta[4] * comp_price
    )


def oracle_price_and_expected_revenue(
    theta, day_sin, day_cos, traffic, comp_price, pmin, pmax
):
    """
    Analytic oracle under linear demand mu(p) = c_t - b*p, b = theta[-1]
    Returns p_star (clipped), expected revenue R_star.
    """
    c_t = compute_c_t(theta, day_sin, day_cos, traffic, comp_price)
    b = theta[-1]  # Correct: b is the positive coefficient
    if b <= 1e-9:
        p_star = 0.5 * (pmin + pmax)
        mu = max(0.0, c_t - b * p_star)
        R_star = p_star * mu
        return p_star, R_star, c_t, b

    # Unconstrained optimal price
    p_star_unconstrained = c_t / (2.0 * b)
    # Clip to bounds
    p_star = float(np.clip(p_star_unconstrained, pmin, pmax))

    # Calculate expected revenue at the (potentially clipped) optimal price
    mu_at_p_star = max(0.0, c_t - b * p_star)
    R_star = p_star * mu_at_p_star

    return p_star, float(R_star), float(c_t), float(b)


# ---------------------------
# Pricing Environment
# ---------------------------
class PricingEnv:
    def __init__(
        self,
        T=T_DEFAULT,
        seed=DEFAULT_SEED,
        theta=None,
        sigma=GAUSSIAN_NOISE_SIGMA,
        price_bounds=PRICE_BOUNDS,
        comp_base=COMP_BASE,
        comp_amp=COMP_AMPLITUDE,
        comp_period=COMP_PERIOD,
        comp_noise_sigma=COMP_NOISE_SIGMA,
        forced_exploration=FORCED_EXPLORATION,
        traffic_mean=TRAFFIC_MEAN,
        traffic_poisson=TRAFFIC_POISSON,
    ):
        self.T = int(T)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.sigma = float(sigma)
        self.price_bounds = tuple(float(x) for x in price_bounds)
        self.comp_base = float(comp_base)
        self.comp_amp = float(comp_amp)
        self.comp_period = float(comp_period)
        self.comp_noise_sigma = float(comp_noise_sigma)
        self.forced_exploration = int(forced_exploration)
        self.traffic_mean = float(traffic_mean)
        self.traffic_poisson = bool(traffic_poisson)

        self.theta = np.array(theta if theta is not None else TRUE_THETA, dtype=float)

        # internal state
        self.t = 0
        self.day_counter = 0
        self.curr_context = None

    def reset(self):
        self.t = 0
        self.day_counter = 0
        # Reset RNG for reproducible runs
        self.rng = np.random.RandomState(self.seed)
        self.curr_context = self._get_context()
        return self.curr_context

    def _sample_competitor_price(self):
        # seasonal: base + amp * sin(2pi * t / period) + noise
        val = (
            self.comp_base
            + self.comp_amp
            * np.sin(2.0 * np.pi * (self.t % self.comp_period) / self.comp_period)
            + self.rng.normal(0.0, self.comp_noise_sigma)
        )
        return float(np.clip(val, self.price_bounds[0], self.price_bounds[1]))

    def _sample_traffic(self):
        if self.traffic_poisson:
            lam = max(
                1.0,
                self.traffic_mean
                + 400.0 * np.sin(2.0 * np.pi * (self.day_counter % 7) / 7.0),
            )
            return float(self.rng.poisson(lam))
        else:
            return float(max(0.0, self.rng.normal(self.traffic_mean, 50.0)))

    def _get_context(self):
        day = self.day_counter % 7
        sin_d, cos_d = cyclical_day_features(day)
        traffic = self._sample_traffic()
        comp_price = self._sample_competitor_price()
        # context vector for agents: [sin, cos, traffic, comp_price]
        return np.array([sin_d, cos_d, traffic, comp_price], dtype=float)

    def step(self, price):
        """
        Accepts a price (float), clamps to bounds, and returns:
            context (for next step),
            demand (observed),
            revenue,
            done (bool),
            info (dict with oracle info)
        """
        price = float(np.clip(price, self.price_bounds[0], self.price_bounds[1]))

        # Use the context the agent acted on
        context_t = self.curr_context
        day = self.day_counter % 7
        sin_d, cos_d, traffic, comp_price = (
            context_t[0],
            context_t[1],
            context_t[2],
            context_t[3],
        )

        # feature phi = [1, sin, cos, traffic, comp_price, -price]
        phi = np.array([1.0, sin_d, cos_d, traffic, comp_price, -price], dtype=float)

        # expected demand mu_t
        mu = float(np.dot(self.theta, phi))

        # observe demand with gaussian noise and clamp to >= 0
        demand = float(max(0.0, mu + self.rng.normal(0.0, self.sigma)))

        revenue = float(price * demand)

        # oracle (expected) price and expected revenue for reporting (uses true theta)
        p_star, R_star, c_t, b = oracle_price_and_expected_revenue(
            self.theta,
            sin_d,
            cos_d,
            traffic,
            comp_price,
            self.price_bounds[0],
            self.price_bounds[1],
        )

        info = {
            "phi": phi,
            "c_t": c_t,
            "b": b,
            "oracle_price": p_star,
            "oracle_expected_revenue": R_star,
            "day": int(day),
            "traffic": float(traffic),
            "competitor_price": float(comp_price),
            "context": context_t,
        }

        # advance time
        self.t += 1
        self.day_counter += 1
        done = self.t >= self.T

        # Get context for the *next* decision
        self.curr_context = self._get_context()

        return self.curr_context, demand, revenue, done, info
# change 6 — Decide quarter-wise representative routes
# change 19 — Implement RL on normalized US Airway data
# change 22 — Removed redundant files
# change 25 — Optimize discount recommendation
# change 27 — Fix missing import
# change 44 — Fix demand bug
# change 49 — Improve config parsing
