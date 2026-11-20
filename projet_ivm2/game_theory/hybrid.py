import numpy as np
from metaheuristic.operators import (
    tournament_index, uniform_crossover, one_point_crossover, order_crossover,
)
from game_theory.game import child_payoff, strategy_payoffs
from game_theory.controller import (
    ReplicatorController, RandomController, FixedController,
)


class HybridGameGA:
    def __init__(self, pop_size=100, n_generations=200,
                 mu_E=0.05, mu_X=0.005,
                 p_c_E=0.9, p_c_X=0.6,
                 elitism=2,
                 x_init=0.5, x_min=0.05, x_max=0.95, eta=4.0,
                 mode="replicator", seed=None):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mu_E, self.mu_X = mu_E, mu_X
        self.p_c_E, self.p_c_X = p_c_E, p_c_X
        self.elitism = elitism
        self.rng = np.random.default_rng(seed)
        ctrl_seed = seed + 1 if seed is not None else None
        kwargs = dict(x_init=x_init, x_min=x_min, x_max=x_max, eta=eta,
                      seed=ctrl_seed)
        controllers = {"replicator": ReplicatorController,
                       "random": RandomController,
                       "fixed": FixedController}
        self.mode = mode
        self.controller = controllers[mode](**kwargs)

    def _crossovers(self, problem):
        if getattr(problem, "problem_type", "continuous") == "permutation":
            return order_crossover, order_crossover
        return uniform_crossover, one_point_crossover  # (E, X)

    def run(self, problem):
        cx_E, cx_X = self._crossovers(problem)
        pop = problem.init_population(self.pop_size, self.rng)
        fitness = problem.evaluate_pop(pop)

        best_idx = int(np.argmin(fitness))
        best_ind, best_fit = pop[best_idx].copy(), float(fitness[best_idx])
        history_best = [best_fit]
        history_xE = []
        n_children = self.pop_size - self.elitism

        for _ in range(self.n_generations):
            # 1. Proportion courante de la stratégie Exploration
            x_E = self.controller.proportion()
            history_xE.append(x_E)

            # 2. Reproduction : chaque couple joue E ou X
            elite_idx = np.argsort(fitness)[:self.elitism]
            children, f_ref, is_E = [], [], []
            while len(children) < n_children:
                explorer = self.rng.random() < x_E
                i1 = tournament_index(fitness, rng=self.rng)
                i2 = tournament_index(fitness, rng=self.rng)
                ref = min(fitness[i1], fitness[i2])
                if explorer:
                    c1, c2 = cx_E(pop[i1], pop[i2], self.p_c_E, rng=self.rng)
                    mu = self.mu_E
                else:
                    c1, c2 = cx_X(pop[i1], pop[i2], self.p_c_X, rng=self.rng)
                    mu = self.mu_X
                for c in (c1, c2):
                    children.append(problem.mutate(c, mu, self.rng))
                    f_ref.append(ref)
                    is_E.append(explorer)
            children = np.array(children[:n_children])
            f_ref = np.array(f_ref[:n_children])
            is_E = np.array(is_E[:n_children])

            # 3. Évaluation et gains des stratégies
            f_children = problem.evaluate_pop(children)
            u = child_payoff(f_ref, f_children)
            u_E, u_X = strategy_payoffs(u, is_E)

            # 4. Nouvelle population, meilleur individu
            pop = np.concatenate([pop[elite_idx], children])
            fitness = np.concatenate([fitness[elite_idx], f_children])
            idx = int(np.argmin(fitness))
            if fitness[idx] < best_fit:
                best_fit, best_ind = float(fitness[idx]), pop[idx].copy()
            history_best.append(best_fit)

            # 5. Mise à jour de la dynamique (réplicateur)
            self.controller.update(u_E, u_X)

        return {"best_individual": best_ind, "best_fitness": best_fit,
                "history": history_best, "history_xE": history_xE}
