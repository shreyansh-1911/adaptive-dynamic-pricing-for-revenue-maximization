import os
from math import pi
import numpy as np

# ---------------------------
# Configuration / Defaults
# ---------------------------
DEFAULT_SEED = 42
T_DEFAULT = 4000 
PRICE_BOUNDS = (1.0, 100.0) 

# Exploration
FORCED_EXPLORATION = 10

# Noise & Context
GAUSSIAN_NOISE_SIGMA = 2.0
TRAFFIC_MEAN = 500
TRAFFIC_POISSON = True
COMP_BASE = 50.0
COMP_AMPLITUDE = 15.0
COMP_PERIOD = 30.0
COMP_NOISE_SIGMA = 5.0

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------
# True Parameters (Legacy support)
# ---------------------------
TRUE_THETA = np.array(
    [10.0, 2.0, -1.0, 0.02, 0.5, 1.0], 
    dtype=float,
)

def cyclical_day_features(day_index):
    """Maps 0..6 to (sin, cos)."""
    angle = 2.0 * pi * (day_index % 7) / 7.0
    return float(np.sin(angle)), float(np.cos(angle))

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
        
        self.t = 0
        self.day_counter = 0
        self.curr_context = None

    def reset(self):
        self.t = 0
        self.day_counter = 0
        self.rng = np.random.RandomState(self.seed)
        self.curr_context = self._get_context()
        return self.curr_context

    def _sample_competitor_price(self):
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
        return float(max(0.0, self.rng.normal(self.traffic_mean, 50.0)))

    def _get_context(self):
        day = self.day_counter % 7
        sin_d, cos_d = cyclical_day_features(day)
        traffic = self._sample_traffic()
        comp_price = self._sample_competitor_price()
        return np.array([sin_d, cos_d, traffic, comp_price], dtype=float)

    def step(self, price):
        price = float(np.clip(price, self.price_bounds[0], self.price_bounds[1]))
        context_t = self.curr_context
        sin_d, cos_d, traffic, comp_price = context_t

        # --- THE TRAP LOGIC (Non-Linear) ---
        is_luxury = traffic > 500
        
        if is_luxury:
            # Luxury Mode: High Price ($85) is optimal
            base_demand = 20.0
            demand_curve = np.exp(-0.5 * ((price - 85.0) / 15.0)**2)
            oracle_price = 85.0
            oracle_rev = 85.0 * 20.0
        else:
            # Budget Mode: Low Price ($25) is optimal
            base_demand = 40.0
            demand_curve = np.exp(-0.5 * ((price - 25.0) / 10.0)**2)
            oracle_price = 25.0
            oracle_rev = 25.0 * 40.0
            
        demand = base_demand * demand_curve
        demand = float(max(0.0, demand + self.rng.normal(0.0, self.sigma)))
        revenue = float(price * demand)

        # Fake linear info to prevent crashing old agents
        phi = np.array([1.0, sin_d, cos_d, traffic, comp_price, -price], dtype=float)
        
        info = {
            "phi": phi,
            "c_t": 0.0,
            "b": 1.0,
            "oracle_price": oracle_price,
            "oracle_expected_revenue": oracle_rev,
            "day": int(self.day_counter % 7),
            "traffic": float(traffic),
            "competitor_price": float(comp_price),
            "context": context_t,
        }

        self.t += 1
        self.day_counter += 1
        done = self.t >= self.T
        self.curr_context = self._get_context()
        
        return self.curr_context, demand, revenue, done, info

    def _is_weekend(self):
        day = self.day_counter % 7
        return 1.0 if day in (5, 6) else 0.0

    def get_agent_context(self):
        sin_d, cos_d, traffic, comp_price = self.curr_context
        weekend = self._is_weekend()
        return np.array([float(traffic), float(comp_price), float(sin_d), float(weekend)], dtype=float)

    def get_context(self):
        return self.get_agent_context()# change 30 — Add pricing UI
# change 31 — Fix reward edge case
# change 47 — Improve exploration policy
