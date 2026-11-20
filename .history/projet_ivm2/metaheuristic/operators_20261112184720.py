"""Opérateurs génétiques de base (sélection, croisements)."""
import numpy as np


def tournament_index(fitness, k=3, rng=None):
    """Sélection par tournoi de taille k (minimisation). Retourne l'INDICE
    du vainqueur (utile pour connaître la fitness des parents)."""
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, len(fitness), size=k)
    return int(idx[np.argmin(fitness[idx])])


def tournament_selection(population, fitness, k=3, rng=None):
    """Sélection par tournoi de taille k. Retourne une copie de l'individu."""
    return population[tournament_index(fitness, k, rng)].copy()


def uniform_crossover(parent1, parent2, p_c, rng=None):
    """Croisement uniforme (fortement diversifiant) avec probabilité p_c."""
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > p_c:
        return parent1.copy(), parent2.copy()
    mask = rng.random(len(parent1)) < 0.5
    return np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)


def one_point_crossover(parent1, parent2, p_c, rng=None):
    """Croisement à un point (plus conservateur : blocs préservés)."""
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > p_c:
        return parent1.copy(), parent2.copy()
    point = int(rng.integers(1, len(parent1)))
    return (np.concatenate([parent1[:point], parent2[point:]]),
            np.concatenate([parent2[:point], parent1[point:]]))


def order_crossover(parent1, parent2, p_c, rng=None):
    """Order Crossover (OX) pour permutations (TSP)."""
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > p_c:
        return parent1.copy(), parent2.copy()
    n = len(parent1)
    i, j = sorted(rng.integers(0, n, size=2))
    if i == j:
        return parent1.copy(), parent2.copy()

    def ox(p1, p2):
        child = -np.ones(n, dtype=int)
        child[i:j + 1] = p1[i:j + 1]
        used = np.zeros(n, dtype=bool)
        used[p1[i:j + 1]] = True
        child[child == -1] = p2[~used[p2]]
        return child

    return ox(parent1, parent2), ox(parent2, parent1)
