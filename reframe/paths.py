"""Camera paths as a cinematographer would move: hold still, pan at a
steady speed, or ease between the two — never drift or wobble.

L1-optimal paths (Grundmann, Kwatra, Essa, "Auto-Directed Video
Stabilization with Robust L1 Optimal Camera Paths", CVPR 2011, as used for
YouTube's stabiliser): minimise the L1 norms of the path's first, second and
third differences. L1 (unlike least squares) makes most of those differences
exactly zero, so the path is made of constant, linear and parabolic pieces:
holds, pans and eases.

Constraints: the subject stays inside a corridor of the crop (soft, heavily
penalised, so a subject too fast for the speed limit bends the path rather
than making it impossible), the crop stays inside the picture, and the speed
is limited.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

W_VELOCITY = 10.0    # weights on velocity, acceleration, jerk (per second; Grundmann et al. use 10, 1, 100 per frame)
W_ACCEL = 1.0
W_JERK = 0.1
W_OUTSIDE = 1000.0   # per unit of the subject leaving the corridor
W_TARGET = 2.0       # per unit away from the preferred position (the corridor's middle), per second:
                     # small, and L1, so noise is ignored, but moves happen when the subject moves, not early


def _diff(n: int, order: int) -> csr_matrix:
    d = eye(n, format="csr")
    for _ in range(order):
        d = d[1:] - d[:-1]
    return csr_matrix(d)


def l1_path(lo: np.ndarray, hi: np.ndarray, bounds, fps: float,
            max_speed: float, target: np.ndarray | None = None) -> np.ndarray | None:
    """Crop position per sample with lo[i] <= p[i] <= hi[i] where possible,
    within `bounds` always (a (min, max) pair, or one pair per sample),
    |speed| <= max_speed (units / s).
    Returns None if the solver fails (callers fall back)."""
    n = len(lo)
    per = list(bounds) if len(bounds) == n and np.ndim(bounds) == 2 else [tuple(bounds)] * n
    b_lo, b_hi = np.array([b[0] for b in per]), np.array([b[1] for b in per])
    if n < 4:
        return np.clip((lo + hi) / 2, b_lo, b_hi)
    # Scale: differences per second, so the weights don't depend on the sample rate.
    D = [_diff(n, k) * fps ** k / fps for k in (1, 2, 3)]
    sizes = [d.shape[0] for d in D]
    # Variables: p (n), e1, e2, e3 (|D_k p| <= e_k), s_lo, s_hi (corridor slack), g (|p - target| <= g).
    nv = n + sum(sizes) + 3 * n
    c = np.concatenate([np.zeros(n)] + [np.full(m, w) for m, w in zip(sizes, (W_VELOCITY, W_ACCEL, W_JERK))]
                       + [np.full(2 * n, W_OUTSIDE), np.full(n, W_TARGET / fps)])
    rows, rhs = [], []
    off = n
    for d, m in zip(D, sizes):
        e = csr_matrix((np.ones(m), (np.arange(m), off + np.arange(m))), shape=(m, nv))
        dp = hstack([d, csr_matrix((m, nv - n))])
        rows += [dp - e, -dp - e]
        rhs += [np.zeros(m), np.zeros(m)]
        off += m
    s_lo = csr_matrix((np.ones(n), (np.arange(n), off + np.arange(n))), shape=(n, nv))
    s_hi = csr_matrix((np.ones(n), (np.arange(n), off + n + np.arange(n))), shape=(n, nv))
    P = hstack([eye(n), csr_matrix((n, nv - n))])
    rows += [-P - s_lo, P - s_hi]                  # p >= lo - s_lo, p <= hi + s_hi
    rhs += [-lo, hi]
    g = csr_matrix((np.ones(n), (np.arange(n), off + 2 * n + np.arange(n))), shape=(n, nv))
    target = (lo + hi) / 2 if target is None else target
    rows += [P - g, -P - g]
    rhs += [target, -target]
    var_bounds = per + [(0, None)] * (nv - n)
    # Speed limit; dropped if the hard bounds move faster than it allows
    # (keeping the face beats the pan speed).
    v = hstack([_diff(n, 1), csr_matrix((n - 1, nv - n))])
    step = max_speed / fps
    for limited in (True, False):
        A = vstack(rows + ([v, -v] if limited else [])).tocsr()
        b = np.concatenate(rhs + ([np.full(n - 1, step), np.full(n - 1, step)] if limited else []))
        res = linprog(c, A_ub=A, b_ub=b, bounds=var_bounds, method="highs")
        if res.status == 0:
            return np.clip(res.x[:n], b_lo, b_hi)
    return None
