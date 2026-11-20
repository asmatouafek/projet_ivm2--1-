"""Problème du sac à dos 0/1 (binaire), avec pénalité de dépassement."""
import numpy as np


class KnapsackProblem:
    problem_type = "binary"

    def __init__(self, n_items=50, capacity=None, seed=42):
        rng = np.random.default_rng(seed)
        self.values = rng.integers(1, 100, size=n_items)
        self.weights = rng.integers(1, 50, size=n_items)
        self.capacity = capacity if capacity else int(0.5 * self.weights.sum())
        self.dim = n_items

    def init_population(self, n, rng):
        return rng.integers(0, 2, size=(n, self.dim))

    def evaluate_pop(self, X):
        X = np.atleast_2d(X)
        w = X @ self.weights
        v = X @ self.values
        return -v + np.maximum(0, w - self.capacity) * 10.0

    def evaluate(self, x):
        return float(self.evaluate_pop(x)[0])

    def mutate(self, ind, mu, rng):
        """Bit-flip : chaque bit est inversé avec proba mu."""
        mask = rng.random(len(ind)) < mu
        return np.logical_xor(ind, mask).astype(int)

    def optimum(self):
        """Optimum exact par programmation dynamique (valeur NÉGATIVE,
        cohérente avec la minimisation)."""
        best = np.zeros(self.capacity + 1, dtype=int)
        for w, v in zip(self.weights, self.values):
            best[w:] = np.maximum(best[w:], best[:self.capacity + 1 - w] + v)
        return -float(best[-1])
