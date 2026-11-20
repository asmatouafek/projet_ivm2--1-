"""Dynamique du réplicateur (jeu à deux stratégies)."""


def replicator_step(x_E, u_E, u_X, eta, x_min=0.0, x_max=1.0):
    """Un pas de la dynamique du réplicateur en temps discret :

        x_E(t+1) = x_E(t) + eta * x_E * (u_E - u_moy)
                 = x_E(t) + eta * x_E * (1 - x_E) * (u_E - u_X)

    où u_moy = x_E*u_E + (1-x_E)*u_X. La proportion de la stratégie qui
    « paie » mieux que la moyenne augmente. Point fixe intérieur : u_E = u_X.
    x_E est ensuite projeté sur [x_min, x_max] (maintien d'un minimum des
    deux stratégies dans la population)."""
    x_new = x_E + eta * x_E * (1.0 - x_E) * (u_E - u_X)
    return min(max(x_new, x_min), x_max)
