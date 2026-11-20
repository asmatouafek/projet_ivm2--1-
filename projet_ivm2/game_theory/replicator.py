def replicator_step(x_E, u_E, u_X, eta, x_min=0.0, x_max=1.0):
    x_new = x_E + eta * x_E * (1.0 - x_E) * (u_E - u_X)
    return min(max(x_new, x_min), x_max)
