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
