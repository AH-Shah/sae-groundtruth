"""Dense-input SAE: Matryoshka BatchTopK SAE on a frozen cell embedding (SAFFRON recipe) + controls + metrics.

Codes from every method are (n x F) matrices scored by the same functions: activation-matching reliability, lineage selectivity,
gene programs (OLS top-50, Hallmark, XMARK chimera), identity kNN, spatial-axis Spearman / OMP.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

DEFAULT = dict(M=None, k=None, matryoshka=True, route="batch", resample_every=0, steps=6000, batch=4096, lr=3e-3, seed=0, noise=.02, aux_k=16, aux_w=1 / 32, dead_steps=300, device="cpu")


# ------------------------------------------------------------------ model
class MSAE(nn.Module):
    def __init__(self, d, M, prefixes, b_init, D_init):
        super().__init__(); self.prefixes = prefixes
        self.b_dec = nn.Parameter(torch.as_tensor(b_init, dtype=torch.float32).clone())
        self.D = nn.Parameter(torch.as_tensor(D_init, dtype=torch.float32).clone())  # M x d, unit rows
        self.W = nn.Parameter(self.D.detach().T.clone()); self.b_enc = nn.Parameter(torch.zeros(M))
        self.register_buffer("thr", torch.tensor(0.0)); self.register_buffer("last", torch.zeros(M)); self.register_buffer("step", torch.tensor(0.0))

    def pre(self, x): return F.relu((x - self.b_dec) @ self.W + self.b_enc)

    def forward(self, x, k=None, route="batch"):
        a = self.pre(x)
        if k is not None and route == "cell":  # per-cell TopK: exactly k per row (train and eval)
            v, i = a.topk(k, 1); z = torch.zeros_like(a).scatter(1, i, v)
        elif k is not None:  # BatchTopK: keep the k * n largest entries of the batch
            cut = a.flatten().topk(k * len(a)).values[-1]; z = a * (a >= cut).to(a.dtype)
        else: z = a * (a > self.thr).to(a.dtype)
        return z, a


def _init(X, M, seed):
    """b_dec = mean; D = k-means++ seeds of centred rows (unit), as rebuild/sae/core.init_model; more latents than seeds -> random unit rows."""
    from sklearn.cluster import kmeans_plusplus
    rng = np.random.default_rng(seed); sub = X[rng.choice(len(X), min(len(X), 20000), replace=False)]; b = sub.mean(0); c = sub - b
    _, idx = kmeans_plusplus(c, M, random_state=seed); D = c[idx]; D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
    return b, D


def fit(X, cfg):
    """X: (n x d) float32 standardised embedding. Returns model (cpu), history."""
    c = {**DEFAULT, **cfg}; d = X.shape[1]; M = c["M"] or 4 * d; k = c["k"] or max(1, round(.06 * M))
    prefixes = [M // 4, M // 2, M] if c["matryoshka"] else [M]
    torch.manual_seed(c["seed"]); rng = np.random.default_rng(c["seed"]); b, D0 = _init(X, M, c["seed"])
    dev = torch.device(c["device"]); m = MSAE(d, M, prefixes, b, D0).to(dev); Xt = torch.as_tensor(X, dtype=torch.float32).to(dev); var = float(Xt.var(0).sum())
    opt = torch.optim.Adam(m.parameters(), lr=c["lr"]); T = c["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 100) * .5 * (1 + math.cos(math.pi * min(t, T) / T)))
    hist = []
    for t in range(T):
        xb = Xt[torch.as_tensor(rng.integers(0, len(Xt), c["batch"]), device=dev)]
        xin = xb + c["noise"] * max(0.0, 1 - t / (.8 * T)) * torch.randn_like(xb)
        z, a = m(xin, k=k, route=c["route"]); loss = 0.0
        for p in prefixes: loss = loss + ((m.b_dec + z[:, :p] @ m.D[:p] - xb) ** 2).sum(1).mean() / var
        loss = loss / len(prefixes)
        with torch.no_grad(): m.last.copy_(torch.where((z > 0).any(0), m.step, m.last))
        dead = (m.step - m.last) > c["dead_steps"]
        if c["aux_w"] > 0 and bool(dead.any()):
            e = (xb - m.b_dec - z @ m.D).detach(); ad = a.masked_fill(~dead.unsqueeze(0), 0); top = ad.topk(min(c["aux_k"], int(dead.sum())), 1)
            za = torch.zeros_like(a).scatter(1, top.indices, top.values); loss = loss + c["aux_w"] * ((za @ m.D - e) ** 2).sum(1).mean() / e.pow(2).sum(1).mean().clamp_min(1e-9)
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            Dn = m.D / m.D.norm(dim=1, keepdim=True).clamp_min(1e-9); m.D.grad -= (m.D.grad * Dn).sum(1, keepdim=True) * Dn
        opt.step(); sched.step()
        with torch.no_grad():
            m.D.div_(m.D.norm(dim=1, keepdim=True).clamp_min(1e-9)); m.step += 1
            R = c["resample_every"]
            if R and t > 0 and t % R == 0 and t < .7 * T and bool(dead.any()):  # resample dead latents onto high-residual cells (SAFFRON: every 10 epochs)
                e = xb - m.b_dec - z @ m.D; w = e.pow(2).sum(1); idx = torch.multinomial(w / w.sum(), int(dead.sum()), replacement=True)
                dn = e[idx] / e[idx].norm(dim=1, keepdim=True).clamp_min(1e-9); di = torch.nonzero(dead).flatten()
                m.D[di] = dn; m.W[:, di] = .2 * dn.T; m.b_enc[di] = 0; m.last[di] = m.step
                for g in opt.param_groups:  # reset Adam moments of the resampled rows
                    for p_ in g["params"]:
                        st = opt.state.get(p_, {})
                        if p_ is m.D and st: st["exp_avg"][di] = 0; st["exp_avg_sq"][di] = 0
                        if p_ is m.W and st: st["exp_avg"][:, di] = 0; st["exp_avg_sq"][:, di] = 0
        if t % 500 == 0 or t == T - 1: hist.append(dict(step=t, loss=float(loss), dead=int(dead.sum())))
    with torch.no_grad():
        if len(Xt) * M <= 2 ** 28:  # exact cut on all rows
            A = torch.cat([m.pre(Xt[i:i + 65536]) for i in range(0, len(Xt), 65536)]); m.thr.copy_(A.flatten().topk(k * len(A)).values[-1])
        else:
            ix = torch.as_tensor(np.random.default_rng(c["seed"]).choice(len(Xt), 200000, replace=False), device=Xt.device)
            A = torch.cat([m.pre(Xt[ix[i:i + 65536]]).cpu() for i in range(0, len(ix), 65536)]).numpy().ravel()
            m.thr.copy_(torch.tensor(float(np.partition(A, -k * 200000)[-k * 200000])))
    m.k = k; m.route = c["route"]; m = m.cpu()
    if dev.type == "mps": torch.mps.empty_cache()
    return m, hist


@torch.no_grad()
def encode(m, X, chunk=65536):
    Xt = torch.as_tensor(X, dtype=torch.float32); kw = dict(k=m.k, route="cell") if getattr(m, "route", "batch") == "cell" else {}
    return np.vstack([m(Xt[i:i + chunk], **kw)[0].numpy() for i in range(0, len(Xt), chunk)])


@torch.no_grad()
def recon_r2(m, X, prefix=None):
    Xt = torch.as_tensor(X, dtype=torch.float32); z, _ = m(Xt); p = prefix or z.shape[1]
    if getattr(m, "route", "batch") == "cell": z = torch.as_tensor(encode(m, X))
    xh = m.b_dec + z[:, :p] @ m.D[:p]; return float(1 - ((xh - Xt) ** 2).sum() / ((Xt - Xt.mean(0)) ** 2).sum())


# ------------------------------------------------------------------ controls
def varimax(L, it=100, tol=1e-6):
    p, k = L.shape; R = np.eye(k); d = 0
    for _ in range(it):
        Lr = L @ R; u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p)); R = u @ vt; d_new = s.sum()
        if d_new < d * (1 + tol): break
        d = d_new
    return R


def split_signed(Z): return np.hstack([np.maximum(Z, 0), np.maximum(-Z, 0)]).astype(np.float32)


def topk_rows(Z, k):
    """sparsified control: keep the k largest entries of each row of a non-negative code (e.g. split_signed varimax), zero the rest."""
    k = int(max(1, min(k, Z.shape[1]))); Z = np.asarray(Z, np.float32); idx = np.argpartition(-Z, k - 1, axis=1)[:, :k]
    out = np.zeros_like(Z); r = np.arange(len(Z))[:, None]; out[r, idx] = Z[r, idx]; return out


def control_codes(kind, X, K, seed=0, Xfit=None, gene_basis=None):
    """codes of a control fitted on Xfit (default X), applied to X. pca: PCA of the embedding (= the input's own axes);
    varimax: PCA scores rotated by varimax of the PCA axes; ica: FastICA; kmeans: one-hot of K clusters."""
    from sklearn.decomposition import PCA, FastICA
    from sklearn.cluster import MiniBatchKMeans
    Xf = X if Xfit is None else Xfit; d = X.shape[1]
    if kind == "pca":
        p = PCA(d, random_state=seed).fit(Xf); return p.transform(X)
    if kind == "varimax":  # rotation of the gene loadings (gene_basis: d x G axes of the input embedding, e.g. the joint PCs)
        p = PCA(d, random_state=seed).fit(Xf); Lg = p.components_.T * np.sqrt(p.explained_variance_)
        if gene_basis is not None: Lg = gene_basis.T @ (p.components_.T * np.sqrt(p.explained_variance_))
        return p.transform(X) @ varimax(Lg)
    if kind == "ica":
        ic = FastICA(d, random_state=seed, max_iter=400, whiten="unit-variance").fit(Xf); return ic.transform(X)
    if kind == "kmeans":
        km = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=8192).fit(Xf); lab = km.predict(X); Z = np.zeros((len(X), K), np.float32); Z[np.arange(len(X)), lab] = 1; return Z
    raise ValueError(kind)


# ------------------------------------------------------------------ metrics
def alive(Z, thr=1e-3): return ((Z != 0).mean(0) >= thr)


def act_match(Za, Zb, r_thr=.7, signed=False, n_max=60000, seed=0):
    """Hungarian match of features by activation correlation on the same cells; share of matched pairs with r >= r_thr, median r."""
    ia, ib = np.flatnonzero(alive(Za)), np.flatnonzero(alive(Zb))
    if len(ia) == 0 or len(ib) == 0: return 0.0, 0.0
    rows = np.random.default_rng(seed).choice(len(Za), min(len(Za), n_max), replace=False)
    A = Za[rows][:, ia].astype(np.float64); B = Zb[rows][:, ib].astype(np.float64)
    A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-12); B = (B - B.mean(0)) / np.maximum(B.std(0), 1e-12); C = A.T @ B / len(A)
    if signed: C = np.abs(C)
    r, c = linear_sum_assignment(-C); v = C[r, c]; return float((v >= r_thr).sum() / min(len(ia), len(ib))), float(np.median(v))


def lineage_selectivity(Z, lin, lins):
    """p_L = mean_L |z| / sum_L' mean_L' |z| (lineage-size neutral); returns (max p, argmax lineage) per alive feature."""
    A = np.abs(Z); M = np.stack([A[lin == l].mean(0) for l in lins], 1); P = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
    return P.max(1), np.array(lins)[P.argmax(1)]


def gene_programs(Z, L, genes, n_max=40000, seed=0, top=50):
    """OLS slope of z-scored genes on z-scored activation (rows = features): top-50 genes per feature (positive direction)."""
    rows = np.random.default_rng(seed).choice(len(Z), min(len(Z), n_max), replace=False); Lz = L[rows]
    Lz = (Lz - Lz.mean(0)) / np.maximum(Lz.std(0), 1e-9); Zs = Z[rows].astype(np.float32); Zs = (Zs - Zs.mean(0)) / np.maximum(Zs.std(0), 1e-9)
    B = Zs.T @ Lz / len(rows); return [list(genes[np.argsort(-b)[:top]]) for b in B], B


def morans_i(Z, pos, slide, k=10, n_max=20000, seed=0):
    """mean over features of Moran's I on a spatial kNN graph, per slide (sub-sampled), averaged over slides."""
    from scipy.spatial import cKDTree
    out = []; rng = np.random.default_rng(seed)
    for s in np.unique(slide):
        ii = np.flatnonzero(slide == s); ii = ii if len(ii) <= n_max else rng.choice(ii, n_max, replace=False)
        nb = cKDTree(pos[ii]).query(pos[ii], k + 1)[1][:, 1:]; Y = Z[ii].astype(np.float64); Y = Y - Y.mean(0); den = (Y ** 2).sum(0)
        num = (Y[:, None, :] * Y[nb]).sum((0, 1)) / k; ok = den > 0; out.append(np.mean(num[ok] / den[ok]) if ok.any() else np.nan)
    return float(np.nanmean(out))


def axis_scores(Z, tau, cv=5, seed=0):
    """SAFFRON .2: max |Spearman(tau, z_m)| over features on cells where z_m != 0 (>= 200 cells); 5-fold OMP R2 with 2 features."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.model_selection import cross_val_score
    best = 0.0
    for m in np.flatnonzero(alive(Z)):
        nz = (Z[:, m] != 0) & np.isfinite(tau)
        if nz.sum() >= 200: best = max(best, abs(spearmanr(Z[nz, m], tau[nz])[0]))
    ok = np.isfinite(tau); A = Z[ok][:, alive(Z)]; A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-9)
    r2 = cross_val_score(OrthogonalMatchingPursuit(n_nonzero_coefs=2), A, tau[ok], cv=cv, scoring="r2").mean()
    return float(best), float(r2)


def consensus(Xfit, X, cfg, seeds=(0, 1, 2, 3, 4), r_thr=.7, min_votes=3):
    """consensus SAE (cNMF analogue, ): fit one model per seed, encode X; a seed-0 feature is kept if it is matched (Hungarian on
    activation correlation, r >= r_thr) in >= min_votes - 1 other seeds; its consensus activation is the mean of the matched activations."""
    Zs = []
    for s in seeds:
        m, _ = fit(Xfit, {**cfg, "seed": s}); Zs.append(encode(m, X))
    ref = Zs[0]; ia = np.flatnonzero(alive(ref)); acc = ref[:, ia].astype(np.float32).copy(); votes = np.ones(len(ia))
    rows = np.random.default_rng(0).choice(len(X), min(len(X), 30000), replace=False)
    A = ref[rows][:, ia].astype(np.float64); A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-12)
    for Z in Zs[1:]:
        ib = np.flatnonzero(alive(Z)); B = Z[rows][:, ib].astype(np.float64); B = (B - B.mean(0)) / np.maximum(B.std(0), 1e-12)
        C = A.T @ B / len(rows); r, c = linear_sum_assignment(-C); ok = C[r, c] >= r_thr
        votes[r[ok]] += 1; acc[:, r[ok]] += Z[:, ib[c[ok]]]
    keep = votes >= min_votes; return (acc[:, keep] / votes[keep]).astype(np.float32), dict(n_ref=len(ia), n_cons=int(keep.sum()))


# ------------------------------------------------------------------ synthetic check (DS1)
def check(seed=3):
    """planted: 8 coarse unit directions (one per cell) + 24 fine directions (2 per cell, sparse), d 32, noise .05.
    Pass: Matryoshka SAE recovers >= 7/8 coarse in its first M/4 block and >= 20/24 fine overall (cos >= .9); two seeds match by
    activation (act_match) >= .7 share; a random code gives ~0."""
    rng = np.random.default_rng(seed); d, n = 32, 40000
    Dc = rng.standard_normal((8, d)); Dc /= np.linalg.norm(Dc, axis=1, keepdims=True); Df = rng.standard_normal((24, d)); Df /= np.linalg.norm(Df, axis=1, keepdims=True)
    c = rng.integers(0, 8, n); X = 3 * Dc[c] * rng.uniform(.8, 1.2, (n, 1))
    for _ in range(2): f = rng.integers(0, 24, n); X += Df[f] * rng.uniform(.5, 1.5, (n, 1))
    X = (X + .05 * rng.standard_normal(X.shape)).astype(np.float32); X /= np.sqrt((X ** 2).mean())
    m0, _ = fit(X, dict(M=64, k=3, seed=0)); m1, _ = fit(X, dict(M=64, k=3, seed=1))
    D = m0.D.detach().numpy(); cc = (np.abs(Dc @ D[:16].T) >= .9).any(1).sum(); cf = (Df @ D.T >= .9).any(1).sum()
    Z0, Z1 = encode(m0, X), encode(m1, X); rep, _ = act_match(Z0, Z1); rnd, _ = act_match(Z0, (rng.random(Z0.shape) < .05).astype(np.float32))
    return dict(coarse=int(cc), fine=int(cf), seed_rep=rep, random_rep=rnd, ok=bool(cc >= 7 and cf >= 20 and rep >= .7 and rnd < .05))


def check_consensus(seed=3):
    """DS2: planted 8 coarse + 24 fine; consensus over 5 seeds on each cell half reproduces between halves more than single fits
    and recovers the planted features. Pass: consensus half match >= .85 and > single, planted recovery >= .9."""
    rng = np.random.default_rng(seed); d, n = 32, 40000
    Dc = rng.standard_normal((8, d)); Dc /= np.linalg.norm(Dc, axis=1, keepdims=True); Df = rng.standard_normal((24, d)); Df /= np.linalg.norm(Df, axis=1, keepdims=True)
    c = rng.integers(0, 8, n); X = 3 * Dc[c] * rng.uniform(.8, 1.2, (n, 1)); Fm = np.zeros((n, 24))
    for _ in range(2):
        f = rng.integers(0, 24, n); a = rng.uniform(.5, 1.5, n); X += Df[f] * a[:, None]; Fm[np.arange(n), f] += a
    X = (X + .05 * rng.standard_normal(X.shape)).astype(np.float32); X /= np.sqrt((X ** 2).mean()); h = rng.permutation(n) < n // 2; cfg = dict(M=64, k=3)
    Za, _ = consensus(X[h], X, cfg, seeds=(0, 1, 2, 3, 4)); Zb, _ = consensus(X[~h], X, cfg, seeds=(10, 11, 12, 13, 14))
    m0, _ = fit(X[h], dict(cfg, seed=0)); m1, _ = fit(X[~h], dict(cfg, seed=10))
    rc, _ = act_match(Za, Zb); rs, _ = act_match(encode(m0, X), encode(m1, X)); T = np.hstack([np.eye(8)[c], Fm]).astype(np.float32); rec, _ = act_match(T, Za)
    return dict(cons=rc, single=rs, recovered=rec, ok=bool(rc >= .85 and rc > rs and rec >= .9))
