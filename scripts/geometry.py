"""Embedding geometry (Supplementary Table S2).
"""
import os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); DATA = Path(os.environ.get("SAEGT_DATA", ROOT / "data"))
from saegt.metrics import LMAP  # noqa: E402


def prep(E):
    E = E - E.mean(0); return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def stats(name, E, lab):
    E = prep(E.astype(np.float64)); ev_ = np.linalg.eigvalsh(np.cov(E.T))[::-1].clip(0); pr = ev_.sum() ** 2 / (ev_ ** 2).sum()
    ok = lab != "OTH"; Ec = E - E.mean(0); mu = {c: Ec[lab == c].mean(0) for c in set(lab[ok])}
    between = sum(((mu[c]) ** 2).sum() * (lab == c).sum() for c in mu) / (Ec[ok] ** 2).sum()
    dim_share = (E.var(0).max() / E.var(0).sum())
    print(f"{name:22s} n {len(E):6d} d {E.shape[1]:5d} PR {pr:6.1f} top1 {ev_[0] / ev_.sum():.2f} top10 {ev_[:10].sum() / ev_.sum():.2f} between-concept R2 {between:.2f} max-dim var share {dim_share:.3f}", flush=True)


z = np.load(DATA / "inputs/LLM_pythia70m_L3.npz"); stats("LLM Pythia70m L3", z["latent"], z["cls"])
z = np.load(DATA / "inputs/pelka_geneformer.npz", allow_pickle=True); mid = z["celltype"].astype(str); lab = np.array([LMAP.get(x, "OTH") for x in mid])
stats("Geneformer", z["emb_Geneformer"], lab if (lab != "OTH").mean() > .5 else mid)
for key in "AB":
    z = np.load(DATA / f"inputs/section_{key}.npz", allow_pickle=True); lab = np.array([LMAP.get(x, "OTH") for x in z["celltype"].astype(str)])
    for name in z["embeddings"]: stats(f"{key}:{name}", z[f"emb_{name}"], lab)
