# pricing_bandit/agents_dnn.py
"""
Neural Contextual Bandit Agent using MC Dropout + Neural Thompson Sampling.
OPTIMIZED FOR STABILITY, FAST CONVERGENCE, AND LONG-TERM REGRET MINIMIZATION.
"""

from typing import Tuple, List
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# -----------------------------
# Utility: FixableDropout
# -----------------------------
class FixableDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
        self._frozen = False
        self._mask = None

    def forward(self, x):
        if not self.training:
            return F.dropout(x, p=self.p, training=False)

        keep_prob = 1.0 - self.p
        if self._frozen:
            if self._mask is None:
                mask_shape = (1, x.shape[-1])
                mask = (torch.rand(mask_shape, device=x.device) < keep_prob).float()
                self._mask = mask / keep_prob
            return x * self._mask
        else:
            return F.dropout(x, p=self.p, training=True)

    def freeze_mask(self):
        self._frozen = True
        self._mask = None

    def unfreeze_mask(self):
        self._frozen = False
        self._mask = None

    def reset_mask(self):
        self.unfreeze_mask()

# -----------------------------
# Bayesian-ish MLP (Optimized)
# -----------------------------
class BayesNet(nn.Module):
    def __init__(self, input_dim=5, hidden_sizes: List[int] = [64, 64], dropout_p=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        self.fixable_dropout_layers = nn.ModuleList()

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            # USE LEAKY RELU: Prevents "dead neurons" during early training
            layers.append(nn.LeakyReLU(0.01)) 
            fd = FixableDropout(p=dropout_p)
            self.fixable_dropout_layers.append(fd)
            layers.append(fd)
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def freeze_masks(self):
        for fd in self.fixable_dropout_layers:
            fd.freeze_mask()

    def unfreeze_masks(self):
        for fd in self.fixable_dropout_layers:
            fd.unfreeze_mask()

    def reset_masks(self):
        for fd in self.fixable_dropout_layers:
            fd.reset_mask()

# -----------------------------
# Replay buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000): # Smaller capacity to focus on recent trends
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, context: np.ndarray, price: float, revenue: float):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (context.astype(np.float32), float(price), float(revenue))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        contexts, prices, revenues = zip(*batch)
        contexts = np.stack(contexts)
        prices = np.array(prices, dtype=np.float32).reshape(-1, 1)
        revenues = np.array(revenues, dtype=np.float32).reshape(-1, 1)
        return contexts, prices, revenues

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Neural Thompson Agent
# -----------------------------
class NeuralThompsonAgent:
    def __init__(
        self,
        context_dim=4,
        price_bounds=(0.1, 100.0),
        hidden_sizes=[64, 64],
        dropout_p=0.1,
        lr=0.01, # Higher LR for bandit problems
        device='cpu',
        replay_capacity=10000,
        batch_size=64,
        train_every=1,
        train_steps=1
    ):
        self.context_dim = context_dim
        self.input_dim = context_dim + 1
        self.price_bounds = price_bounds
        self.device = torch.device(device)
        
        self.model = BayesNet(input_dim=self.input_dim, hidden_sizes=hidden_sizes, dropout_p=dropout_p).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5) # Weight decay helps generalization
        
        # --- NEW: LEARNING RATE SCHEDULER ---
        # Decays LR by 0.5 every 1000 steps. This helps the model "settle" and reduces regret over time.
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)
        # ------------------------------------

        self.replay = ReplayBuffer(capacity=replay_capacity)
        
        self.batch_size = batch_size
        self.train_every = train_every
        self.train_steps = train_steps
        self.steps = 0
        
        # Statistics for Normalization
        self.ctx_mean = np.zeros(context_dim)
        self.ctx_std = np.ones(context_dim)
        self.n_samples = 0
        
        # Huber Loss is more robust to outliers than MSE
        self.criterion = nn.HuberLoss(delta=10.0)

    def _update_stats(self, context):
        self.n_samples += 1
        if self.n_samples == 1:
            self.ctx_mean = context.copy()
            self.M2 = np.zeros_like(context)
        else:
            delta = context - self.ctx_mean
            self.ctx_mean += delta / self.n_samples
            self.M2 += delta * (context - self.ctx_mean)

    def get_std(self):
        if self.n_samples < 2:
            return np.ones_like(self.ctx_mean)
        return np.sqrt(self.M2 / (self.n_samples - 1))

    def normalize(self, context):
        # Only normalize context, NOT price (price is handled explicitly)
        std = self.get_std()
        std[std < 1e-5] = 1.0
        return (context - self.ctx_mean) / std

    def select_price(self, context: np.ndarray, n_asp_steps: int = 20, asp_lr: float = 0.5) -> float:
        self.model.train()
        self.model.freeze_masks()

        ctx = torch.tensor(context, dtype=torch.float32, device=self.device).view(1, -1)
        low, high = self.price_bounds

        # RANDOM RESTART: Critical for non-convex optimization
        # Try 3 different starting points and pick the best one
        best_price = float(np.random.uniform(low, high))
        best_pred_rev = -float('inf')

        start_points = [
            random.uniform(low, high),
            random.uniform(low, high), 
            (low+high)/2
        ]

        for p0 in start_points:
            price_var = torch.tensor([[p0]], dtype=torch.float32, device=self.device, requires_grad=True)
            opt_input = optim.Adam([price_var], lr=asp_lr)

            for _ in range(n_asp_steps):
                opt_input.zero_grad()
                inp = torch.cat([ctx, price_var], dim=1)
                out = self.model(inp)
                loss = -out.mean()
                loss.backward()
                opt_input.step()
                with torch.no_grad():
                    price_var.clamp_(low, high)
            
            # Check if this restart found a better peak
            with torch.no_grad():
                final_rev = self.model(torch.cat([ctx, price_var], dim=1)).item()
                if final_rev > best_pred_rev:
                    best_pred_rev = final_rev
                    best_price = price_var.item()

        self.model.unfreeze_masks()
        return best_price

    def store_transition(self, context: np.ndarray, price: float, revenue: float):
        # SCALING REWARD: Neural Nets hate large numbers like 10,000. 
        # We scale revenue down for training stability.
        scaled_revenue = revenue / 100.0 
        self.replay.push(context, price, scaled_revenue)
        self.steps += 1

        if len(self.replay) >= self.batch_size and (self.steps % self.train_every == 0):
            for _ in range(self.train_steps):
                self._train_step()
            
            # Step the Scheduler every time we train
            self.scheduler.step()

    def _train_step(self):
        self.model.train()
        contexts, prices, revenues = self.replay.sample(self.batch_size)
        
        inp = torch.tensor(np.concatenate([contexts, prices], axis=1), dtype=torch.float32, device=self.device)
        tgt = torch.tensor(revenues.reshape(-1), dtype=torch.float32, device=self.device)
        
        preds = self.model(inp)
        loss = self.criterion(preds, tgt)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()# change 1 — Initial commit
# change 11 — Implement Q-learning Agent
# change 23 — Merge PR #2
# change 39 — Add training plots
# change 51 — Add XGBoost implementation
# change 52 — feat: added real world demonstration of models
