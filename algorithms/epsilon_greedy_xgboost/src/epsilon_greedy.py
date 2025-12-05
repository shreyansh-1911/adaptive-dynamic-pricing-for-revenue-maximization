# src/epsilon_greedy.py
import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)    # how many times each arm was chosen
        self.values = np.zeros(n_arms)    # average reward (revenue) for each arm

    def select_arm(self):
        """
        Choose which price (arm) to try:
        - Explore with probability ε
        - Exploit (best arm) otherwise
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        """
        Update running average of reward for the chosen arm
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n