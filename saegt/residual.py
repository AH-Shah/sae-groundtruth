"""Two-stage residual consensus SAE. Stage 1 = consensus of plain BatchTopK fits; its decoder is a ridge LS fit on
the centred consensus codes; stage 2 = SAE (single or consensus) on the rescaled residual. Codes are returned for the fit rows and for all rows."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from saegt import sae as dense


def consensus2(Xf, X, cfg, seeds=(0, 1, 2, 3, 4), r_thr=.7, min_votes=3):
    """dense.consensus with codes on both the fit rows Xf and all rows X (one set of models); same keep rule."""
    Zf, Za = [], []
    for s in seeds:
        m, _ = dense.fit(Xf, {**cfg, "seed": s}); Zf.append(dense.encode(m, Xf)); Za.append(dense.encode(m, X))
    ref = Zf[0]; ia = np.flatnonzero(dense.alive(ref)); votes = np.ones(len(ia)); accf = ref[:, ia].copy(); acca = Za[0][:, ia].copy()
    rows = np.random.default_rng(0).choice(len(Xf), min(len(Xf), 30000), replace=False)
    A = ref[rows][:, ia].astype(np.float64); A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-12)
    for zf, za in zip(Zf[1:], Za[1:]):
        ib = np.flatnonzero(dense.alive(zf)); B = zf[rows][:, ib].astype(np.float64); B = (B - B.mean(0)) / np.maximum(B.std(0), 1e-12)
        C = A.T @ B / len(rows); r, c = linear_sum_assignment(-C); ok = C[r, c] >= r_thr
        votes[r[ok]] += 1; accf[:, r[ok]] += zf[:, ib[c[ok]]]; acca[:, r[ok]] += za[:, ib[c[ok]]]
    keep = votes >= min_votes
    return (accf[:, keep] / votes[keep]).astype(np.float32), (acca[:, keep] / votes[keep]).astype(np.float32)


def ls_residual(Zf, Xf, Za, Xa):
    """Ridge LS decoder on the fit rows' centred codes; residuals of fit rows and all rows (fit-row centring), variance share rho."""
    zm, xm = Zf.mean(0), Xf.mean(0); Zc = (Zf - zm).astype(np.float64); G = Zc.T @ Zc
    lam = 1e-6 * np.trace(G) / max(len(G), 1); D = np.linalg.solve(G + lam * np.eye(len(G)), Zc.T @ (Xf - xm))
    Rf = (Xf - xm) - Zc @ D; Ra = (Xa - xm) - (Za - zm) @ D; rho = float((Rf ** 2).sum() / ((Xf - xm) ** 2).sum())
    return Rf.astype(np.float32), Ra.astype(np.float32), D, rho


_CACHE = {}


def two_stage(Xf, X, cfg, frac=.25, stage2="cons", seeds=(0, 1, 2, 3, 4)):
    """returns codes on all rows: [Z1, Z2], Z2, info. cfg = stage-1 dense cfg (M, k, steps, device, matryoshka False)."""
    key = (Xf.shape, float(Xf[:3].sum()), X.shape, tuple(seeds), cfg["M"], cfg["k"])  # stage 1 shared across stage-2 variants in one process
    if key not in _CACHE: _CACHE[key] = consensus2(Xf, X, cfg, seeds)
    Z1f, Z1a = _CACHE[key]
    Rf, Ra, _, rho = ls_residual(Z1f, Xf, Z1a, X); sc = np.sqrt((Rf ** 2).mean()); Rf, Ra = Rf / sc, Ra / sc
    M2 = max(2, int(round(frac * cfg["M"]))); c2 = {**cfg, "M": M2, "k": max(1, round(M2 / 16))}
    if stage2 == "cons": _, Z2 = consensus2(Rf, Ra, c2, tuple(s + 100 for s in seeds))
    else: Z2 = dense.encode(dense.fit(Rf, {**c2, "seed": seeds[0] + 100})[0], Ra)
    return np.hstack([Z1a, Z2]), Z2, dict(rho=rho, n1=Z1a.shape[1], n2=Z2.shape[1])
