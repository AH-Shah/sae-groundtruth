"""SAE vs rotations on the spatial (A, B) and Geneformer (P) embeddings.

usage: python scripts/run_embeddings.py A|B|P METHODS [INPUTS]
  METHODS: comma list of tsae_cons, tsae, msae_cons, msae, pca, varimax, ica, varimax_topk, tsae_res<pct>[s|only]
           (varimax_topk needs a consensus arm earlier in the same list: it is cut to that consensus's mean L0)
  env: SAEGT_SEED (0, 1, 2), SAEGT_M (dictionary size override, e.g. 256 for P), SAEGT_STEPS (3000), SAEGT_DATA (data), SAEGT_OUT (results/runs)
paper settings: A/B: tsae_cons,tsae,varimax_topk and msae_cons,msae,pca,varimax,ica; P: the same with SAEGT_M=256 for the SAE arms.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from saegt import metrics as mt, residual, sae as dense  # noqa: E402

DATA = Path(os.environ.get("SAEGT_DATA", ROOT / "data")); OUT = Path(os.environ.get("SAEGT_OUT", ROOT / "results/runs")); OUT.mkdir(parents=True, exist_ok=True)
SEED = int(os.environ.get("SAEGT_SEED", "0")); STEPS = int(os.environ.get("SAEGT_STEPS", "3000")); MFIX = os.environ.get("SAEGT_M", "")
FILES = {"A": "section_A.npz", "B": "section_B.npz", "P": "pelka_geneformer.npz"}


def load(key):
    """counts (cells x genes, CSR), genes, cell-type labels ('' = unlabelled or confidence < .7), raw embeddings {name: array}."""
    z = np.load(DATA / "inputs" / FILES[key], allow_pickle=True); cf = DATA / "built" / f"counts_{key}.npz"
    if not cf.exists(): raise SystemExit(f"{cf} missing: run scripts/build_inputs.py {key} <source> first (counts stay at their public source)")
    c = np.load(cf); C = sp.csr_matrix((c["data"], c["indices"], c["indptr"]), shape=tuple(c["shape"])).astype(np.float32)
    emb = {k[4:]: z[k].astype(np.float32) for k in z.files if k.startswith("emb_")}
    return C, z["genes"].astype(str), z["celltype"].astype(str), emb


def main(key, methods, only=None, tag=""):
    tag = tag + (f"_seed{SEED}" if SEED else "")
    import torch
    torch.set_num_threads(8); mk = json.load(open(DATA / "inputs/markers.json"))["cosmx2000"]; hm = mt.hallmark(DATA / "built/hallmark.json")
    C, genes, lab, emb = load(key); rows = np.arange(C.shape[0])
    lib = np.asarray(C.sum(1)).ravel(); L = C.multiply(1e4 / np.maximum(lib, 1)[:, None]).tocsr(); L.data = np.log1p(L.data); L = L.toarray().astype(np.float32)
    lin = np.array([mt.LMAP.get(x, "OTH") for x in lab]); half = np.random.default_rng(1 + 100 * SEED).permutation(len(rows)) < len(rows) // 2
    res, feats_all = [], []
    for name, E in emb.items():
        if only and name not in only: continue
        E = E - E.mean(0); E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)  # prep "unit"
        E = (E / np.sqrt((E ** 2).mean())).astype(np.float32); d = E.shape[1]; M = int(MFIX) if MFIX else 4 * d; k = max(1, round(.0625 * M))
        run_l0 = {}
        for method in methods:
            t0 = time.time()

            def fit(Xf, s):
                s = s + 3 * SEED
                dev = "mps" if torch.backends.mps.is_available() else "cpu"
                base = dict(M=M, k=k, steps=STEPS, route="batch", resample_every=0, device=dev)
                if method == "msae_cons":
                    Zc, inf = dense.consensus(Xf, E, base, seeds=tuple(range(10 * s, 10 * s + 5))); return Zc, inf
                if method == "msae":
                    m, h = dense.fit(Xf, {**base, "seed": s}); return dense.encode(m, E), dict(r2=dense.recon_r2(m, E), dead_train=h[-1]["dead"])
                if method.startswith("tsae_res"):  # two-stage residual model (Methods)
                    tg = method[8:]; frac = int("".join(ch for ch in tg if ch.isdigit())) / 100
                    Zc, Z2, inf = residual.two_stage(Xf, E, {**base, "matryoshka": False}, frac=frac, stage2="single" if tg.endswith("s") else "cons", seeds=tuple(range(10 * s, 10 * s + 5)))
                    return (Z2 if tg.endswith("only") else Zc), inf
                if method == "tsae_cons":  # consensus of plain BatchTopK fits (main SAE of the paper)
                    return dense.consensus(Xf, E, {**base, "matryoshka": False}, seeds=tuple(range(10 * s, 10 * s + 5)))
                if method == "tsae":  # single plain BatchTopK fit
                    m, h = dense.fit(Xf, {**base, "seed": s, "matryoshka": False}); return dense.encode(m, E), dict(r2=dense.recon_r2(m, E), dead_train=h[-1]["dead"])
                if method == "varimax_topk":  # varimax codes sparsified to the consensus SAE's mean L0 in this run
                    if name not in run_l0: raise SystemExit("varimax_topk needs a consensus arm (tsae_cons or msae_cons) earlier in METHODS")
                    return dense.topk_rows(dense.split_signed(dense.control_codes("varimax", E, 0, seed=s, Xfit=Xf)), run_l0[name]), {}
                return dense.split_signed(dense.control_codes(method, E, 0, seed=s, Xfit=Xf)), {}
            Z, info = fit(E, 0); al = dense.alive(Z); Za = Z[:, al]
            if method in ("tsae_cons", "msae_cons"): run_l0[name] = int(round(float((Z != 0).sum(1).mean())))
            ok = lin != "OTH"; pmax, plin = dense.lineage_selectivity(Za[ok], lin[ok], mt.LINS); tops, _ = dense.gene_programs(Za, L, genes)
            hq = np.array([mt.enrich(t, genes, hm)[1] for t in tops]); chim = np.array([mt.foreign_markers(";".join(t), l) >= 3 for t, l in zip(tops, plin)])
            chim_ref = np.array([mt.ref_foreign(t, l, mk) >= 3 for t, l in zip(tops, plin)])
            ii = lab != ""; knn, pur = mt.bench_lineage(Z[ii], lab[ii])
            Zh, _ = fit(E[half], 1); Zg, _ = fit(E[~half], 2); rep, rmed = dense.act_match(Zh, Zg)
            r = dict(slide=key, input=name, method=method, prep="unit", route="batch", resample=0, d=d, M=Z.shape[1], k=k if method in ("msae", "tsae") else np.nan, alive=int(al.sum()), L0=float((Z != 0).sum(1).mean()),
                     R1_rep=rep, R1_rmed=rmed, knn=knn, purity=pur, mono=float((pmax >= .7).mean()), mono_clean=int(((pmax >= .7) & ~chim).sum()),
                     hallmark=float((hq < .05).mean()), chimeric=float(chim.mean()), chimeric_ref=float(chim_ref.mean()), n_mono=int((pmax >= .7).sum()), sec=0.0, seed=SEED, **info)
            r["sec"] = time.time() - t0; res.append(r)
            feats_all.append(pd.DataFrame(dict(slide=key, input=name, method=method, feature=np.flatnonzero(al), use=(Za != 0).mean(0), pmax=pmax, lineage=plin,
                                               hallmark_q=hq, chimeric=chim, top10=[";".join(t[:10]) for t in tops])))
            print(f"{key} {name} {method} {r['sec']:.0f}s R1 {rep:.2f} knn {knn:.3f} mono {r['mono']:.2f} hallmark {r['hallmark']:.2f}", flush=True)
            pd.DataFrame(res).to_csv(OUT / f"T2_{key}_unit_batch_rs0{tag}.csv", index=False); pd.concat(feats_all).to_csv(OUT / f"T2_{key}_unit_batch_rs0{tag}_features.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2: raise SystemExit(__doc__)
    only = a[2].split(",") if len(a) > 2 else None
    main(a[0], tuple(a[1].split(",")), only=only, tag="_" + a[1].replace(",", "+") + ("_" + a[2].replace(",", "+") if only else "") + (f"_M{MFIX}" if MFIX else ""))
