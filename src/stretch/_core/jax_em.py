"""Whole EM loop of FHMM_ParentSet.fit in one XLA call, for one hidden parent (the default path).

Same maths as _forward_backward + _maximization_step, except that only sum_t xi_t is formed
(never the (T, N, N) tensor) and the weighted least squares is solved via normal equations in
the same standardised basis instead of stacking a (T*N, p) design for sklearn.
Callers must run it under `jax.enable_x64(True)`.
"""
import math

import jax
import jax.numpy as jnp
from jax import lax

LN2 = math.log(2)


def _estep(Zc, Zh, y, phi, sigma, A, pi):
    pc = Zc.shape[1]
    mu = (Zc @ phi[:pc])[:, None] + (Zh @ phi[pc:]).T  # (T, N)
    s = jnp.maximum(sigma, 1e-9)
    logE = -0.5 * jnp.log2(2 * jnp.pi * s) - 0.5 * ((y[:, None] - mu) ** 2 / s) / LN2
    E = jnp.maximum(jnp.exp2(jnp.clip(logE, -700, None)), 1e-300)
    N = E.shape[1]
    a0 = pi * E[0]
    s0 = a0.sum()
    a0 = jnp.where(s0 == 0, 1.0 / N, a0)
    s0 = jnp.where(s0 == 0, 1.0, s0)
    a0 = a0 / s0

    def fwd(a_prev, e):
        a = (a_prev @ A) * e
        sc = a.sum()
        a = jnp.where(sc == 0, a_prev, a / sc)
        return a, (a, jnp.where(sc == 0, 0.0, jnp.log2(sc)))

    _, (alpha_rest, lls) = lax.scan(fwd, a0, E[1:])
    alpha = jnp.vstack([a0[None], alpha_rest])
    ll = jnp.log2(s0) + lls.sum()

    def bwd(b_next, e_next):
        b = A @ (e_next * b_next)
        sc = b.sum()
        b = jnp.where(sc > 0, b / sc, 1.0 / N)
        return b, b

    _, beta_rev = lax.scan(bwd, jnp.ones(N), E[1:][::-1])
    beta = jnp.vstack([beta_rev[::-1], jnp.ones((1, N))])
    gamma = alpha * beta
    gamma = gamma / gamma.sum(1, keepdims=True)
    # sum_t xi_t = A * sum_t outer(alpha_t / d_t, E_{t+1} beta_{t+1}); uninformative steps fall back to A / sum(A)
    eb = E[1:] * beta[1:]
    d = ((alpha[:-1] @ A) * eb).sum(1)
    ok = d > 1e-12
    S = jnp.where(ok[:, None], alpha[:-1] / jnp.where(ok, d, 1.0)[:, None], 0.0).T @ eb
    xi = A * S + (~ok).sum() * A / A.sum()
    return ll, gamma, xi


def _mstep(Zc, Zh, y, gamma, xi, mask, pi):
    T, pc = Zc.shape
    N, _, k = Zh.shape
    p = pc + k
    w = gamma.sum(1)
    H = jnp.einsum("ts,stk->tk", gamma, Zh)
    Wz = Zh * gamma.T[:, :, None]
    G = jnp.zeros((p, p))
    G = G.at[:pc, :pc].set((Zc * w[:, None]).T @ Zc)
    G = G.at[:pc, pc:].set(Zc.T @ H).at[pc:, :pc].set(H.T @ Zc)
    G = G.at[pc:, pc:].set(Wz.reshape(-1, k).T @ Zh.reshape(-1, k))
    b = jnp.concatenate([Zc.T @ (w * y), H.T @ y])
    # Standardise columns 1.. with their weighted mean/std, like upstream, then map back.
    W = w.sum()
    mu = G[0, 1:] / W
    sd = jnp.sqrt(jnp.maximum(jnp.diag(G)[1:] / W - mu ** 2, 0))
    sd = jnp.where(sd < 1e-9, 1.0, sd)
    Mt = jnp.eye(p).at[0, 1:].set(-mu / sd).at[1:, 1:].set(jnp.diag(1 / sd))
    phi = Mt @ jnp.linalg.lstsq(Mt.T @ G @ Mt, Mt.T @ b)[0]
    mu_all = (Zc @ phi[:pc])[:, None] + (Zh @ phi[pc:]).T
    sigma = jnp.maximum(jnp.sum(gamma * (y[:, None] - mu_all) ** 2) / T, 1e-6)
    An = (xi + 1.0) * mask
    rs = An.sum(1, keepdims=True)
    A = jnp.where(rs != 0, An / jnp.where(rs != 0, rs, 1.0), 1.0 / N)
    return phi, sigma, A, pi / pi.sum(), An  # pi is never re-estimated upstream either


def em(Zc, Zh, y, phi, sigma, A, pi, mask, tol, max_iter):
    def cond(c):
        return (c[6] < max_iter) & ~c[7]

    def body(c):
        phi, sigma, A, pi, prev, _, it, _, _ = c
        ll, gamma, xi = _estep(Zc, Zh, y, phi, sigma, A, pi)
        phi, sigma, A, pi, An = _mstep(Zc, Zh, y, gamma, xi, mask, pi)
        return phi, sigma, A, pi, ll, ll, it + 1, jnp.abs(ll - prev) < tol, An

    c = lax.while_loop(cond, body, (phi, sigma, A, pi, -jnp.inf, -jnp.inf, 0, False, A))
    return dict(phi=c[0], sigma=c[1], A=c[2], pi=c[3], ll=c[5], A_counts=c[8])


em_jit = jax.jit(em, static_argnames="max_iter")
