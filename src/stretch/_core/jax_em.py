"""FHMM_ParentSet.fit's EM loop as one XLA call, for the default path (one hidden parent, no lasso, no pruning).

E-step: gamma and sum_t xi_t are gradients of the forward log-likelihood (gamma = E dll/dE,
xi = A dll/dA), so there is no backward pass and no (T, N, N) xi tensor.
M-step: upstream's weighted least squares on the stacked (N*T, p) design, without sklearn.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np


def _loglik(E, A, pi):  # scaled forward pass, natural log
    def step(a, e):
        a = (a @ A) * e
        return a / a.sum(), jnp.log(a.sum())

    a0 = pi * E[0]
    _, logs = jax.lax.scan(step, a0 / a0.sum(), E[1:])
    return jnp.log(a0.sum()) + logs.sum()


@jax.jit
def _em(Z, y, phi, sigma, A, pi, mask, tol, max_iter):  # Z: (N states, T, p)
    def body(c):
        phi, sigma, A, _, ll, _, it = c
        s = jnp.maximum(sigma, 1e-9)
        E = jnp.maximum(jnp.exp(-0.5 * (jnp.log(2 * jnp.pi * s) + (y - Z @ phi) ** 2 / s)), 2.0 ** -700).T
        new_ll, (dE, dA) = jax.value_and_grad(_loglik, (0, 1))(E, A, pi)
        gamma, xi = jnp.maximum(E * dE, 0), A * dA  # clamp: roundoff gives -1e-17 on impossible states
        w = jnp.sqrt(gamma.T)[..., None]
        X, Y = (w * Z).reshape(-1, Z.shape[2]), (w[..., 0] * y).ravel()
        phi = jnp.linalg.solve(X.T @ X, X.T @ Y)  # ponytail: normal equations square the condition number; lstsq if fits go unstable (1.8x slower)
        sigma = jnp.maximum(jnp.sum(gamma.T * (y - Z @ phi) ** 2) / len(y), 1e-6)
        counts = (xi + 1.0) * mask
        new_ll = new_ll / math.log(2)
        return phi, sigma, counts / counts.sum(1, keepdims=True), counts, new_ll, jnp.abs(new_ll - ll) < tol, it + 1

    return jax.lax.while_loop(lambda c: (c[6] < max_iter) & ~c[5], body, (phi, sigma, A, A, -jnp.inf, False, 0))


def fit(m):
    fixed, (name,) = m.get_ols_mapping()
    p = m.parents[name]
    Zc = np.column_stack([np.ones_like(m.Y_eff), m.Y_prev] + [m.parents[n]["Z_fixed_path"] for n in fixed])
    Z = np.stack([np.hstack([Zc, z]) for z in p["Z"].values()])
    phi = np.concatenate([m.base_phi] + [m.parents[n]["phi"] for n in fixed + [name]])
    with jax.enable_x64(True):
        r = _em(Z, m.Y_eff, phi, m.Sigma[0], m.A_joint, p["pi"] / p["pi"].sum(), p["mask"], m.tol, m.max_iter)
        phi, sigma, p["A"], p["A_counts"], ll = (np.asarray(v) for v in r[:5])
    m.scatter_phi(phi)
    m.Sigma = sigma[None]
    m.A_joint = m._calculate_joint_transition_matrix()
    if any(not np.any(d["phi"]) for d in m.parents.values()):
        return -np.inf
    return float(ll) + sum(m.parents[f]["fixed_ll"] for f in m.fixed_parent_names)
