"""Composant théorie des jeux : gains (payoffs) des deux stratégies.

Jeu évolutionnaire à deux stratégies pures :
    E = Exploration  (croisement diversifiant, forte mutation)
    X = eXploitation (croisement conservateur, faible mutation)

Gain d'un enfant produit par la stratégie s :
    u = 1 si f_enfant < f_ref, 0 sinon,
avec f_ref = meilleure fitness des deux parents (minimisation).
Le gain d'une stratégie est la moyenne de u sur les enfants qu'elle a produits,
c.-à-d. son TAUX DE RÉUSSITE : il est naturellement normalisé dans [0, 1] et
indépendant de l'échelle du problème (Rastrigin ~ 90, sac à dos ~ -2000).
"""
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
