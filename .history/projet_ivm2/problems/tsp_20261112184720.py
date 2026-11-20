"""Problème du voyageur de commerce (TSP) + parseur TSPLIB."""
import numpy as np


def parse_tsplib(path):
    coords = []
    with open(path, "r") as f:
        in_section = False
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                in_section = True
                continue
            if line.startswith("EOF"):
                break
            if in_section and line:
                parts = line.split()
                coords.append((float(parts[1]), float(parts[2])))
    return np.array(coords)


class TSPProblem:
    problem_type = "permutation"

    def __init__(self, coords):
        self.coords = coords
        self.n = len(coords)
        diff = coords[:, None, :] - coords[None, :, :]
        self.dist = np.linalg.norm(diff, axis=2)

    def init_population(self, n, rng):
        return np.array([rng.permutation(self.n) for _ in range(n)])

    def evaluate_pop(self, T):
        T = np.atleast_2d(T)
        return self.dist[T, np.roll(T, -1, axis=1)].sum(axis=1)

    def evaluate(self, tour):
        return float(self.evaluate_pop(tour)[0])

    def mutate(self, tour, mu, rng):
        """Mutation par échanges : chaque position est échangée avec une
        position aléatoire avec probabilité mu (mu*n échanges en moyenne)."""
        tour = tour.copy()
        n_swaps = rng.binomial(self.n, mu)
        for _ in range(n_swaps):
            i, j = rng.integers(0, self.n, size=2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour
