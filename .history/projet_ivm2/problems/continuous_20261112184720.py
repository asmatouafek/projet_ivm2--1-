"""Fonctions de benchmark continues (Rastrigin, Rosenbrock)."""
import numpy as np


class ContinuousProblem:
    problem_type = "continuous"

    def __init__(self, func_name="rastrigin", dim=10, bounds=(-5.12, 5.12)):
        self.name = func_name
        self.dim = dim
        self.bounds = bounds

    def init_population(self, n, rng):
        lo, hi = self.bounds
        return rng.uniform(lo, hi, size=(n, self.dim))

    def evaluate_pop(self, X):
        """Évaluation vectorisée d'une population (n, dim)."""
        X = np.atleast_2d(X)
        if self.name == "rastrigin":
            return 10 * self.dim + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)
        if self.name == "rosenbrock":
            return np.sum(100 * (X[:, 1:] - X[:, :-1]**2)**2
                          + (1 - X[:, :-1])**2, axis=1)
        raise ValueError(self.name)

    def evaluate(self, x):
        return float(self.evaluate_pop(x)[0])

    def mutate(self, ind, mu, rng):
        """Mutation gaussienne : chaque gène est perturbé avec proba mu."""
        mask = rng.random(len(ind)) < mu
        noise = rng.normal(0, 0.1, size=len(ind))
        lo, hi = self.bounds
        return np.clip(ind + mask * noise, lo, hi)
