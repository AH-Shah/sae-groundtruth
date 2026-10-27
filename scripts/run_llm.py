"""SAEs vs rotations on the Pythia-70m embedding (fit on 900k tokens, metrics on 100k held-out tokens).
env: SAEGT_SEED, SAEGT_ARMS, SAEGT_L0_FROM
"""
import os, sys, time, numpy as np, pandas as pd, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("SAEGT_DATA", ROOT / "data")); RUNS = Path(os.environ.get("SAEGT_OUT", ROOT / "results/runs")); RUNS.mkdir(parents=True, exist_ok=True)
from saegt import sae as dense  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
OUT = str(RUNS) + "/"; SEED = int(os.environ.get("SAEGT_SEED", "0"))
ARMS = os.environ.get("SAEGT_ARMS", "tsae_cons,tsae,pca,varimax,ica,varimax_topk").split(","); L0_FROM = os.environ.get("SAEGT_L0_FROM", "tsae_cons")
z = np.load(DATA / "inputs/LLM_pythia70m_L3_1M.npz"); X = z["latent"].astype(np.float32); cl_all = z["cls"]
X -= X.mean(0); X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9); X *= 1 / np.sqrt((X ** 2).mean())
rng = np.random.default_rng(0); ev_idx = np.sort(rng.choice(len(X), 100000, replace=False)); fit_mask = np.ones(len(X), bool); fit_mask[ev_idx] = False
EV, cl = X[ev_idx], cl_all[ev_idx]; XF = X[fit_mask]; del X
n, d = XF.shape; M = min(4 * d, n // 50); k = max(1, round(M / 16))
LINS = sorted(set(cl) - {"OTH"}); ok = cl != "OTH"; half = np.random.default_rng(1 + 100 * SEED).permutation(n) < n // 2
dev = "mps" if torch.backends.mps.is_available() else "cpu"; l0 = {}
rot_sub = lambda Xf: Xf[np.random.default_rng(7).choice(len(Xf), min(len(Xf), 200000), replace=False)]


def fit(method, Xf, s):
    s = s + 3 * SEED; cfg = dict(M=M, k=k, steps=3000, device=dev)
    if method == "tsae_cl": return dense.consensus_codes_cl(dense.fits_codes(Xf, EV, {**cfg, "matryoshka": False}, seeds=tuple(range(10 * s, 10 * s + 5))))[0]
    if method == "tsae_cons": return dense.consensus(Xf, EV, {**cfg, "matryoshka": False}, seeds=tuple(range(10 * s, 10 * s + 5)))[0]
    if method == "msae_cons": return dense.consensus(Xf, EV, cfg, seeds=tuple(range(10 * s, 10 * s + 5)))[0]
    if method == "tsae": return dense.encode(dense.fit(Xf, {**cfg, "seed": s, "matryoshka": False})[0], EV)
    if method == "msae": return dense.encode(dense.fit(Xf, {**cfg, "seed": s})[0], EV)
    if method == "varimax_topk": return dense.topk_rows(dense.split_signed(dense.control_codes("varimax", EV, 0, seed=s, Xfit=rot_sub(Xf))), l0["k"])
    return dense.split_signed(dense.control_codes(method, EV, 0, seed=s, Xfit=rot_sub(Xf)))


rows = []; sub = np.random.default_rng(0).choice(np.flatnonzero(ok), 20000, replace=False)
for method in ARMS:
    t0 = time.time(); Z = fit(method, XF, 0); al = dense.alive(Z); Za = Z[:, al]
    if method == L0_FROM: l0["k"] = int(round(float((Z != 0).sum(1).mean())))
    pmax, plin = dense.lineage_selectivity(Za[ok], cl[ok], LINS)
    knn = balanced_accuracy_score(cl[sub], cross_val_predict(KNeighborsClassifier(15), Z[sub], cl[sub], cv=5))
    rep, rmed = dense.act_match(fit(method, XF[half], 1), fit(method, XF[~half], 2))
    rows.append(dict(slide="LLMbig", input="Pythia70m_L3_1M", method=method, d=d, M=M, k=k, n_fit=n, alive=int(al.sum()), L0=float((Z != 0).sum(1).mean()), R1_rep=rep, R1_rmed=rmed,
                     knn=knn, mono=float((pmax >= .7).mean()), n_mono=int((pmax >= .7).sum()), n_cls=len(set(plin[pmax >= .7])), sec=time.time() - t0, seed=SEED))
    pd.DataFrame(rows).to_csv(OUT + f"T2_LLMbig_{'+'.join(ARMS)}{f'_seed{SEED}' if SEED else ''}.csv", index=False)
    print(f"{method} {rows[-1]['sec']:.0f}s R1 {rep:.2f} knn {knn:.3f} mono {rows[-1]['mono']:.2f} n_mono {rows[-1]['n_mono']} alive {rows[-1]['alive']} L0 {rows[-1]['L0']:.1f}", flush=True)
