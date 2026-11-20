import numpy as np


def child_payoff(f_ref, f_child):
    """Gain 0/1 de chaque enfant : a-t-il strictement battu son meilleur parent ?"""
    return (np.asarray(f_child) < np.asarray(f_ref)).astype(float)


def strategy_payoffs(u, is_explorer):
    """Gain moyen (u_E, u_X) = taux de réussite de chaque stratégie.
    Une stratégie sans enfant reçoit le gain moyen de la population
    (neutre : n'influence pas la dynamique)."""
    u = np.asarray(u, dtype=float)
    is_explorer = np.asarray(is_explorer, dtype=bool)
    u_all = u.mean()
    u_E = u[is_explorer].mean() if is_explorer.any() else u_all
    u_X = u[~is_explorer].mean() if (~is_explorer).any() else u_all
    return float(u_E), float(u_X)
