"""[Ported from antig/N2/tools/77_geneformer_embed.py by tools/97_port_scripts.py: paths; lineage map inlined (paper: run 1500)]
Bio-embedding arm for the SAE paper (rows 306-308): Geneformer V2 (cached ctheodoris/Geneformer, 18 layers, d 1152) cell embeddings
of the Pelka 2021 CRC scRNA reference (independent author labels ClusterMidway -> 9 lineages via tools/72 MAP).
Tokenisation as Geneformer: counts / cell total * 1e4 / gene median (gc104M), rank non-zero genes, top 2046 (V1 length, MPS memory), <cls> ... <eos>;
cell embedding = mean of the penultimate layer over gene tokens (Geneformer default emb_layer -1).

usage: 77_geneformer_embed.py profile           -> timing on 32 cells
       77_geneformer_embed.py run [per_lin 2500] -> trials/dsae/GF_pelka.npz (latent, barcodes, lineage, midway)
"""
from __future__ import annotations

import pickle
import sys
import time
import importlib.util as ilu
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import os
ROOT = Path(__file__).resolve().parents[1]; OUT = Path(os.environ.get("SAEGT_DATA", ROOT / "data")) / "inputs"
PELKA = os.environ.get("PELKA_H5AD", "pelka.h5ad")  # Pelka et al. 2021 atlas (h5ad with raw counts, obs ClusterMidway, var feature_name)
HUB = Path.home() / ".cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots"


def lineage_map():
    return {'EpiT': 'EPI', 'Epi': 'EPI', 'Fibro': 'FIB', 'SmoothMuscle': 'MUR', 'Peri': 'PERI', 'Endo': 'END', 'Macro': 'MYE', 'DC': 'MYE', 'Mono': 'MYE', 'Granulo': 'MYE', 'Mast': 'MAST', 'Plasma': 'BPL', 'B': 'BPL', 'TCD4': 'TNK', 'TCD8': 'TNK', 'NK': 'TNK', 'ILC': 'TNK', 'TZBTB16': 'TNK', 'Tgd': 'TNK'}  # ClusterMidway -> 9 lineages (research tools/72 MAP, inlined)


def files():
    model = next(p for p in HUB.iterdir() if (p / "model.safetensors").exists())
    tok = next(HUB.glob("*/geneformer/token_dictionary_gc104M.pkl")); med = next(HUB.glob("*/geneformer/gene_median_dictionary_gc104M.pkl"))
    return model, pickle.load(open(tok, "rb")), pickle.load(open(med, "rb"))


def sample(per_lin, seed=0):
    import anndata as ad
    a = ad.read_h5ad(PELKA, backed="r"); MAP = lineage_map()
    lab = np.array([MAP.get(x, "") for x in a.obs["ClusterMidway"].astype(str).values]); rng = np.random.default_rng(seed); idx = []
    for l in sorted(set(lab) - {""}):
        w = np.flatnonzero(lab == l); idx.append(rng.choice(w, min(per_lin, len(w)), replace=False))
    idx = np.sort(np.concatenate(idx))
    X = a.raw.X[idx] if a.raw is not None else a.X[idx]; X = sp.csr_matrix(X).astype(np.float32)
    return X, np.asarray(a.var_names).astype(str), a.obs_names.values[idx].astype(str), lab[idx], a.obs["ClusterMidway"].astype(str).values[idx]


def tokenize(X, ens, tokd, medd, L=2048):  # 2048 (V1 length): 4096 x bs 8 exceeds MPS memory
    keep = np.array([g in tokd and g in medd for g in ens]); gi = np.flatnonzero(keep)
    tok = np.array([tokd[g] for g in ens[gi]]); med = np.array([medd[g] for g in ens[gi]], np.float32)
    Xn = X[:, gi].multiply(1e4 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)[:, None]).tocsr().multiply(1 / med[None]).tocsr()
    out = []
    for i in range(Xn.shape[0]):
        s, e = Xn.indptr[i], Xn.indptr[i + 1]; o = np.argsort(-Xn.data[s:e])[: L - 2]
        out.append(np.concatenate([[tokd["<cls>"]], tok[Xn.indices[s:e][o]], [tokd["<eos>"]]]))
    return out


def embed(seqs, model_dir, bs=2):
    import torch
    from transformers import BertForMaskedLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = BertForMaskedLM.from_pretrained(model_dir, output_hidden_states=True).eval().to(dev); E = []
    order = np.argsort([len(s) for s in seqs]); res = [None] * len(seqs)
    with torch.no_grad():
        for b in range(0, len(seqs), bs):
            ii = order[b:b + bs]; T = max(len(seqs[i]) for i in ii); ids = np.zeros((len(ii), T), np.int64); att = np.zeros_like(ids)
            for r, i in enumerate(ii): ids[r, :len(seqs[i])] = seqs[i]; att[r, :len(seqs[i])] = 1
            ids_t, att_t = torch.as_tensor(ids, device=dev), torch.as_tensor(att, device=dev)
            h = m(input_ids=ids_t, attention_mask=att_t).hidden_states[-2]  # penultimate layer
            g = att_t.clone(); g[:, 0] = 0; g[torch.arange(len(ii)), att_t.sum(1) - 1] = 0  # gene tokens only
            v = (h * g[..., None]).sum(1) / g.sum(1, keepdim=True)
            for r, i in enumerate(ii): res[i] = v[r].float().cpu().numpy()
            if dev == "mps": torch.mps.empty_cache()
    return np.stack(res).astype(np.float32)


def main(mode, per_lin=2500):
    model_dir, tokd, medd = files()
    X, ens, bc, lab, mid = sample(4 if mode == "profile" else per_lin)
    t0 = time.time(); seqs = tokenize(X, ens, tokd, medd); t1 = time.time(); Z = embed(seqs, model_dir); t2 = time.time()
    print(f"cells {len(bc)} median tokens {int(np.median([len(s) for s in seqs]))} tokenize {t1 - t0:.1f}s embed {t2 - t1:.1f}s "
          f"({(t2 - t1) / len(bc):.2f} s/cell) d {Z.shape[1]} lineages {dict(zip(*np.unique(lab, return_counts=True)))}", flush=True)
    if mode == "run": np.savez(OUT / "GF_pelka.npz", latent=Z, barcodes=bc, lineage=lab, midway=mid)  # data/inputs/pelka_geneformer.npz adds the panel counts (research tools/96)


if __name__ == "__main__":
    a = sys.argv[1:]; main(a[0], *(int(x) for x in a[1:2]))
