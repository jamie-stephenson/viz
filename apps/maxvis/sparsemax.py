from __future__ import annotations
import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.shape[-1] != 3:
        raise ValueError('softmax expects last dimension of size 3')
    max_logits = np.max(logits, axis=-1, keepdims=True)
    stabilized = logits - max_logits
    exp_vals = np.exp(stabilized)
    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
    return exp_vals / sum_exp

def _sparsemax_3d_batch(z: np.ndarray) -> np.ndarray:
    z_sorted = np.sort(z, axis=1)[:, ::-1]
    cumsum = np.cumsum(z_sorted, axis=1)
    ks = np.arange(1, 4, dtype=z.dtype)[None, :]
    cond = z_sorted > (cumsum - 1) / ks
    k_max = np.sum(cond, axis=1)
    idx = (k_max - 1).astype(int).reshape(-1, 1)
    s_k = np.take_along_axis(cumsum, idx, axis=1).squeeze(1)
    tau = (s_k - 1) / k_max
    p = z - tau[:, None]
    p = np.maximum(p, 0.0)
    p[np.isclose(p, 0.0, atol=1e-12)] = 0.0
    sums = np.sum(p, axis=1, keepdims=True)
    near_one = np.isclose(sums, 1.0, atol=1e-12)
    if not np.all(near_one):
        p = np.divide(p, sums, out=p, where=sums != 0)
    return p

def sparsemax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.shape[-1] != 3:
        raise ValueError('sparsemax expects last dimension of size 3')
    orig_shape = logits.shape
    flat = logits.reshape(-1, 3)
    out = _sparsemax_3d_batch(flat)
    return out.reshape(orig_shape)

def entmax_alpha(logits: np.ndarray, alpha: float, n_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """
    Compute the alpha-entmax mapping for 3D logits.

    For alpha=1 it reduces to softmax; for alpha=2 it reduces to sparsemax.
    For 1 < alpha < 2, it solves for tau via bisection such that
        sum((z_i - tau)_+ ** (1/(alpha-1))) = 1
    and returns p_i = ((z_i - tau)_+) ** (1/(alpha-1)).
    The result is normalized to sum to 1 for numerical stability.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.shape[-1] != 3:
        raise ValueError('entmax_alpha expects last dimension of size 3')
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    if np.isclose(alpha, 1.0):
        return softmax(logits)
    if np.isclose(alpha, 2.0):
        return sparsemax(logits)
    if not (1.0 < alpha < 2.0):
        raise ValueError('alpha-entmax supported for 1 < alpha < 2 (or exactly 1, 2)')

    power = 1.0 / (alpha - 1.0)
    orig_shape = logits.shape
    flat = logits.reshape(-1, 3)
    out = np.empty_like(flat)

    for i in range(flat.shape[0]):
        z = flat[i]
        # Bracket tau so that S(lo) > 1 and S(hi) < 1
        z_min = float(np.min(z))
        z_max = float(np.max(z))
        lo = z_min - 10.0
        hi = z_max
        # Ensure the lower bound yields S>1
        def S(tau: float) -> float:
            return float(np.sum(np.clip(z - tau, 0.0, None) ** power))
        # Expand lower bound if needed (very rare with the chosen offset)
        s_lo = S(lo)
        attempts = 0
        while s_lo <= 1.0 and attempts < 10:
            lo -= 10.0
            s_lo = S(lo)
            attempts += 1

        # Bisection to find tau
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            s_mid = S(mid)
            if s_mid > 1.0:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break
        tau = hi
        p = np.clip(z - tau, 0.0, None) ** power
        sum_p = float(np.sum(p))
        if sum_p > 0.0:
            p = p / sum_p
        # Clean very small values to exact zero for display stability
        p[np.isclose(p, 0.0, atol=1e-12)] = 0.0
        out[i] = p

    return out.reshape(orig_shape)
