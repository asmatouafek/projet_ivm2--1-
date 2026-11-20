import numpy as np
from game_theory.replicator import replicator_step


class ReplicatorController:
    def __init__(self, x_init=0.5, x_min=0.05, x_max=0.95, eta=4.0, seed=None):
        self.x = x_init
        self.x_min, self.x_max, self.eta = x_min, x_max, eta

    def proportion(self):
        return self.x

    def update(self, u_E, u_X):
        self.x = replicator_step(self.x, u_E, u_X, self.eta,
                                 self.x_min, self.x_max)


class RandomController:
    def __init__(self, x_min=0.05, x_max=0.95, seed=None, **_):
        self.x_min, self.x_max = x_min, x_max
        self.rng = np.random.default_rng(seed)

    def proportion(self):
        return float(self.rng.uniform(self.x_min, self.x_max))

    def update(self, u_E, u_X):
        pass


class FixedController:
    def __init__(self, x_init=0.5, **_):
        self.x = x_init

    def proportion(self):
        return self.x

    def update(self, u_E, u_X):
        pass
