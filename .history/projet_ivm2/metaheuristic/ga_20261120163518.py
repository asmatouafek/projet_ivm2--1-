import numpy as np
from metaheuristic.operators import (
    tournament_index, uniform_crossover, one_point_crossover, order_crossover,
)

class GeneticAlgorithm:
    def __init__(self, pop_size=100, n_generations=200,
                 p_c=0.8, mu=0.05, elitism=2,
                 crossover_fn="auto", seed=None):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.p_c = p_c
        self.mu = mu
        self.elitism = elitism
        self.crossover_fn = crossover_fn
        self.rng = np.random.default_rng(seed)

    def _pick_crossover(self, problem):
        if self.crossover_fn == "auto":
            if getattr(problem, "problem_type", "continuous") == "permutation":
                return order_crossover
            return uniform_crossover
        mapping = {"uniform": uniform_crossover,
                   "one_point": one_point_crossover,
                   "order": order_crossover}
        return mapping[self.crossover_fn]

    def run(self, problem):
        crossover = self._pick_crossover(problem)
        pop = problem.init_population(self.pop_size, self.rng)
        fitness = problem.evaluate_pop(pop)
        best_idx = int(np.argmin(fitness))
        best_ind, best_fit = pop[best_idx].copy(), float(fitness[best_idx])
        history = [best_fit]
        n_children = self.pop_size - self.elitism

        for _ in range(self.n_generations):
            elite_idx = np.argsort(fitness)[:self.elitism]
            children = []
            while len(children) < n_children:
                p1 = pop[tournament_index(fitness, rng=self.rng)]
                p2 = pop[tournament_index(fitness, rng=self.rng)]
                c1, c2 = crossover(p1, p2, self.p_c, rng=self.rng)
                children.append(problem.mutate(c1, self.mu, self.rng))
                children.append(problem.mutate(c2, self.mu, self.rng))
            children = np.array(children[:n_children])
            pop = np.concatenate([pop[elite_idx], children])
            fitness = problem.evaluate_pop(pop)
            idx = int(np.argmin(fitness))
            if fitness[idx] < best_fit:
                best_fit, best_ind = float(fitness[idx]), pop[idx].copy()
            history.append(best_fit)

        return {"best_individual": best_ind, "best_fitness": best_fit,
                "history": history}
