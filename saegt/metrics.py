"""Evaluation metrics used in the paper: kNN identity, Hallmark enrichment, chimeric gene programs."""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, rankdata

LMAP = {'EpiT': 'EPI', 'Epi': 'EPI', 'Fibro': 'FIB', 'Peri': 'MUR', 'SmoothMuscle': 'MUR', 'Endo': 'END', 'Plasma': 'BPL', 'B': 'BPL', 'Macro': 'MYE', 'DC': 'MYE', 'Mono': 'MYE', 'Mast': 'MYE', 'Granulo': 'MYE', 'TCD4': 'TNK', 'TCD8': 'TNK', 'NK': 'TNK', 'ILC': 'TNK', 'TZBTB16': 'TNK', 'Tgd': 'TNK'}
LINS = ['EPI', 'FIB', 'MYE', 'END', 'MUR', 'BPL', 'TNK']
REFM = {'MUR': ['MUR', 'PERI'], 'MYE': ['MYE', 'MAST']}
XMARK = {'EPI': 'EPCAM KRT8 KRT18 KRT19 KRT20 CDH1 CEACAM5 CEACAM6 MUC2 TFF3 SPINK4 REG4 CLCA1 FCGBP LGALS4 PIGR FABP1 PHGR1 ELF3 CLDN4', 'TNK': 'CD3E CD3D CD2 TRAC NKG7 GNLY CD247 CD8A IL7R CCL5 GZMA GZMK', 'BPL': 'MS4A1 CD79A CD79B MZB1 JCHAIN IGKC IGHA1 IGHG1 IGHM IGLC2 TNFRSF17 DERL3', 'MYE': 'CD68 CD163 CD14 C1QA C1QB C1QC CSF1R AIF1 FCER1G TYROBP LAPTM5 MS4A6A S100A8 S100A9', 'FIB': 'COL1A1 COL1A2 COL3A1 DCN LUM PDGFRA FAP COL6A3 SFRP2 MMP2 COL5A1 FBLN1', 'END': 'PECAM1 VWF CDH5 CLDN5 PLVAP KDR EMCN ACKR1 RAMP2', 'MUR': 'ACTA2 MYH11 DES RGS5 NOTCH3 CNN1 ACTG2'}


def unit(Z):
    return (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)).astype(np.float32)


def knn(Uq, Ur, k, self_pos=None, seed=0, chunk=4096):
    """Cosine kNN of query rows against reference rows; reference order permuted (random tie-breaking); self excluded via self_pos."""
    perm = np.random.default_rng(seed).permutation(len(Ur)); inv = np.argsort(perm)
    R = torch.from_numpy(np.ascontiguousarray(Ur[perm], np.float32)); Q = torch.from_numpy(np.ascontiguousarray(Uq, np.float32))
    out = np.empty((len(Q), k), np.int64); ties = 0
    for i in range(0, len(Q), chunk):
        s = Q[i:i + chunk] @ R.T
        if self_pos is not None:
            sp_ = self_pos[i:i + chunk]; ok = sp_ >= 0  # -1 = query not in the reference
            s[torch.arange(s.shape[0])[torch.from_numpy(ok)], torch.from_numpy(inv[sp_[ok]])] = -np.inf
        v, ix = s.topk(k + 1, dim=1); ties += int(((v[:, k - 1] - v[:, k]).abs() < 1e-6).sum()); out[i:i + chunk] = perm[ix[:, :k].numpy()]
    return out, ties / max(len(Q), 1)


def vote(lab, nn, K):
    n, k = nn.shape
    return np.bincount(np.repeat(np.arange(n) * K, k) + lab[nn].ravel(), minlength=n * K).reshape(n, K).argmax(1)


def bench_lineage(Z, lab, seed=0):
    """lineage-restricted basic bench: kNN macro recall of ClusterMidway types (>= 5 cells) and 15-NN label purity (1 - leak)."""
    ok = lab != ""
    if ok.sum() < 200: return np.nan, np.nan
    U = unit(Z[ok].astype(np.float32)); y = pd.factorize(lab[ok])[0]; Kt = y.max() + 1
    nn, _ = knn(U, U, 15, self_pos=np.arange(len(U)), seed=seed); pred = vote(y, nn, Kt)
    rec = [np.mean(pred[y == t] == t) for t in range(Kt) if (y == t).sum() >= 5]
    return float(np.mean(rec)), float((y[nn] == y[:, None]).mean())


def enrich(top, bg, hm):
    bg = set(bg); n = len(top); res = []
    for k, v in hm.items():
        s = set(v) & bg
        if len(s) < 5: continue
        o = len(set(top) & s); res.append((k, o, hypergeom.sf(o - 1, len(bg), len(s), n)))
    if not res: return "", 1.0, 0
    p = np.array([r[2] for r in res]); q = np.minimum(1, p * len(p) / rankdata(p)); i = int(np.argmin(p)); return res[i][0], float(q[i]), res[i][1]


def foreign_markers(top, lin):
    t = set(top.split(";")); return sum(len(t & set(v.split())) for k, v in XMARK.items() if k != lin)


def ref_foreign(top, lin, mk):
    """chimera with reference-derived markers: count markers of other lineages."""
    t = set(top); own = set(REFM.get(lin, [lin]))
    return sum(len(t & set(v["seed"] + v["eval"])) for k, v in mk.items() if k not in own)


def hallmark(path):
    """MSigDB Hallmark 2020 gene sets (Enrichr library), cached as JSON."""
    return json.load(open(path))
