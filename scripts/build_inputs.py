"""Rebuild the count matrices and the Hallmark gene sets from their public sources into data/built/, and verify each count matrix against
the SHA-256 recorded when the paper's analyses ran (data/inputs/counts_sha256.json). The Zenodo deposit contains only our derived data
(embeddings, cell IDs, cell types, gene list); third-party data stay at their sources.

usage:
  python scripts/build_inputs.py A [232.h5ad]          section A: Crowell et al., results record 10.5281/zenodo.15574384 (downloaded if no path)
  python scripts/build_inputs.py B S0_exprMat_file     section B: CosMx Human Whole Transcriptome Colon Dataset (Bruker Spatial Biology),
                                                       flat file S0_exprMat_file (~20 GB; read in chunks)
  python scripts/build_inputs.py P pelka.h5ad          Pelka et al. 2021 atlas (GEO GSE178341), CZ CELLxGENE h5ad with raw counts
  python scripts/build_inputs.py hallmark              MSigDB Hallmark 2020 gene sets from Enrichr
"""
import hashlib, json, os, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]; DATA = Path(os.environ.get("SAEGT_DATA", ROOT / "data")); BUILT = DATA / "built"; BUILT.mkdir(parents=True, exist_ok=True)
ZENODO_232 = "https://zenodo.org/records/15574384/files/232.h5ad?download=1"
ENRICHR = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
FILES = {"A": "section_A.npz", "B": "section_B.npz", "P": "pelka_geneformer.npz"}


def csr_sha(C):
    C = sp.csr_matrix(C, dtype=np.float32, copy=True); C.sort_indices(); h = hashlib.sha256()
    for a in (C.data.astype(np.float32), C.indices.astype(np.int32), C.indptr.astype(np.int64), np.array(C.shape, np.int64)): h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def save(key, C):
    C = sp.csr_matrix(C, dtype=np.float32, copy=True); C.sort_indices()
    want = json.load(open(DATA / "inputs/counts_sha256.json"))[key]["sha256"]; got = csr_sha(C)
    if got != want: raise SystemExit(f"[{key}] counts differ from the paper's (sha256 {got[:12]} vs {want[:12]}): check the source file version")
    np.savez_compressed(BUILT / f"counts_{key}.npz", data=C.data, indices=C.indices, indptr=C.indptr, shape=np.array(C.shape))
    assert csr_sha(sp.csr_matrix((C.data, C.indices, C.indptr), shape=C.shape)) == want; print(f"[{key}] {C.shape} verified (sha256 {got[:12]}) -> {BUILT}/counts_{key}.npz")


def ids(key):
    z = np.load(DATA / "inputs" / FILES[key], allow_pickle=True); return z["barcodes"].astype(str), z["genes"].astype(str)


def build_A(path=None):
    import anndata as ad
    if path is None:
        path = BUILT / "232.h5ad"
        if not path.exists(): print("downloading 232.h5ad (227 MB) from Zenodo 10.5281/zenodo.15574384"); urllib.request.urlretrieve(ZENODO_232, path)
    bc, genes = ids("A"); a = ad.read_h5ad(path, backed="r")
    ri = pd.Index(a.obs_names.astype(str)).get_indexer(bc); gi = pd.Index(a.var_names.astype(str)).get_indexer(genes); assert (ri >= 0).all() and (gi >= 0).all()
    o = np.argsort(ri); X = sp.csr_matrix(a.X[ri[o]])[np.argsort(o)][:, gi]; save("A", X)


def build_B(path):
    bc, genes = ids("B"); pos = pd.Series(np.arange(len(bc)), index=bc); rows, idx = [], []
    for ch in pd.read_csv(path, usecols=["fov", "cell_ID", *genes], chunksize=20000):
        b = ch.fov.astype(str) + "_" + ch.cell_ID.astype(str); m = b.isin(pos.index).values
        if m.any(): rows.append(sp.csr_matrix(ch.loc[m, list(genes)].values.astype(np.float32))); idx.append(pos[b[m]].values)
    idx = np.concatenate(idx); assert len(idx) == len(bc) and len(set(idx)) == len(bc), "cells missing from the flat file"
    save("B", sp.vstack(rows).tocsr()[np.argsort(idx)])


def build_P(path):
    import anndata as ad
    bc, genes = ids("P"); a = ad.read_h5ad(path, backed="r"); ix = pd.Index(a.obs_names.astype(str)).get_indexer(bc); assert (ix >= 0).all()
    sym = a.var["feature_name"].astype(str).values; panel = set(ids("A")[1]); gi = np.flatnonzero(np.isin(sym, list(panel))); assert (sym[gi] == genes).all()
    X = a.raw.X[ix] if a.raw is not None else a.X[ix]; save("P", sp.csr_matrix(X)[:, gi])


def build_hallmark():
    t = urllib.request.urlopen(ENRICHR, timeout=60).read().decode()
    d = {l.split("\t")[0]: [g for g in l.split("\t")[2:] if g] for l in t.strip().split("\n")}; json.dump(d, open(BUILT / "hallmark.json", "w"))
    print(f"hallmark: {len(d)} sets -> {BUILT}/hallmark.json (Enrichr may have updated the library since the paper; Hallmark scores can differ slightly)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: raise SystemExit(__doc__)
    {"A": lambda: build_A(Path(a[1]) if len(a) > 1 else None), "B": lambda: build_B(a[1]), "P": lambda: build_P(a[1]), "hallmark": build_hallmark}[a[0]]()
