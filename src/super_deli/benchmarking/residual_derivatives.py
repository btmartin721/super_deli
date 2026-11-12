"""Independent NumPy residual forward/backward equations; no Torch or model forward."""

import numpy as np


def evaluate(x, y, weights, *, differentiate=False):
    w = weights
    z = x @ w["input_weight"].T + w["input_bias"]
    h, cache = np.where(z > 0, z, 0), []
    for j in (0, 1):
        u = h @ w[f"block_{j}_first_weight"].T + w[f"block_{j}_first_bias"]
        a = np.where(u > 0, u, 0)
        v = h + a @ w[f"block_{j}_second_weight"].T + w[f"block_{j}_second_bias"]
        cache.append((h, u, a, v))
        h = np.where(v > 0, v, 0)
    logits = h @ w["output_weight"].T + w["output_bias"]
    maxima = np.max(logits, axis=1)
    normalizers = np.log(np.sum(np.exp(logits - maxima[:, None]), axis=1))
    loss = np.mean(maxima + normalizers - logits[np.arange(len(y)), y])
    p = np.exp(logits - (maxima + normalizers)[:, None])
    if not differentiate:
        return float(loss), p
    derivative = p.copy()
    derivative[np.arange(len(y)), y] -= 1
    derivative /= len(y)
    g = {"output_weight": derivative.T @ h, "output_bias": derivative.sum(0)}
    dh = derivative @ w["output_weight"]
    for j in (1, 0):
        old, u, a, v = cache[j]
        dv = dh * (v > 0)
        g[f"block_{j}_second_weight"] = dv.T @ a
        g[f"block_{j}_second_bias"] = dv.sum(0)
        du = (dv @ w[f"block_{j}_second_weight"]) * (u > 0)
        g[f"block_{j}_first_weight"] = du.T @ old
        g[f"block_{j}_first_bias"] = du.sum(0)
        dh = dv + du @ w[f"block_{j}_first_weight"]
    dz = dh * (z > 0)
    g.update(input_weight=dz.T @ x, input_bias=dz.sum(0))
    return float(loss), p, g
