"""Fine cell types (Supplementary Table S3): coverage of the reference fine cell types by the consensus SAE, a single SAE and rotations.
Per embedding and seed s: five plain BatchTopK fits (seeds 30s..30s+4) -> symmetric consensus; the first of the five fits is the single SAE;
PCA, varimax and ICA (split into positive and negative parts); sparse varimax at the consensus L0.
Scores: lineage coverage (of 7); fine coverage = fine types (>= 50 cells) with >= 1 feature whose activity share in one fine type is >= .7;
within-lineage coverage = subtypes that receive >= .7 of a lineage-specific feature's activity among the cells of its lineage.
The paper ran the 50-dimensional embeddings and Geneformer on CPU and the 256-dimensional embeddings on the Apple GPU (SAEGT_DEVICE=mps).
usage: python scripts/run_fine_types.py run [seeds] [A:Expr50,B:CellCharter,...]  -> results/runs/FINE_TYPES.csv
       python scripts/run_fine_types.py table [csv]                                -> Table S3 (markdown)"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from saegt import sae as dense, metrics as mt
DATA = Path(os.environ.get("SAEGT_DATA", ROOT / "data")); OUT = Path(os.environ.get("SAEGT_OUT", ROOT / "results/runs")); OUT.mkdir(parents=True, exist_ok=True)
DEV = os.environ.get("SAEGT_DEVICE", "cpu"); torch.set_num_threads(int(os.environ.get("SAEGT_THREADS", "4")))
FILES = {"A": "section_A.npz", "B": "section_B.npz", "P": "pelka_geneformer.npz"}
TARGETS = ["A:Expr50", "A:CellCharter", "B:Expr50", "B:CellCharter", "B:CADAST", "P:Geneformer"]
ARMS = ["consensus", "single", "pca", "varimax", "ica", "sparse_varimax"]


def load(key, name):
    z = np.load(DATA / "inputs" / FILES[key], allow_pickle=True); E = z[f"emb_{name}"]; lab = z["celltype"].astype(str)
    E = E - E.mean(0); E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9); E = (E / np.sqrt((E ** 2).mean())).astype(np.float32)
    return E, lab, np.array([mt.LMAP.get(x, "OTH") for x in lab])


def scores(Z, lab, lin, fine):
    Za = Z[:, dense.alive(Z)]; ok = lin != "OTH"
    pl, ll = dense.lineage_selectivity(Za[ok], lin[ok], mt.LINS); lin_cov = len(set(ll[pl >= .7]))
    okf = np.isin(lab, fine); pf, lf = dense.lineage_selectivity(Za[okf], lab[okf], fine); fine_cov = len(set(lf[pf >= .7]))
    within = set()
    for L in mt.LINS:
        subs = [f for f in fine if mt.LMAP.get(f) == L]
        if len(subs) < 2: continue
        idx = np.flatnonzero((pl >= .7) & (ll == L)); m = np.isin(lab, subs)
        if not len(idx) or m.sum() < 100: continue
        ps, ls = dense.lineage_selectivity(Za[m][:, idx], lab[m], subs); within |= set(ls[ps >= .7])
    return dict(n_feat=Za.shape[1], lin_cov=lin_cov, fine_cov=fine_cov, within_cov=len(within))


def run(seeds, targets):
    rows = []; out = OUT / "FINE_TYPES.csv"
    for tgt in targets:
        key, name = tgt.split(":"); E, lab, lin = load(key, name); d = E.shape[1]; M = 256 if key == "P" else 4 * d; k = max(1, round(.0625 * M))
        vc = pd.Series(lab[lin != "OTH"]).value_counts(); fine = sorted(vc[vc >= 50].index); subl = {L: [f for f in fine if mt.LMAP.get(f) == L] for L in mt.LINS}
        for s in seeds:
            t0 = time.time(); cfg = dict(M=M, k=k, steps=3000, matryoshka=False, device=DEV)
            Zs = dense.fits_codes(E, E, cfg, seeds=tuple(range(30 * s, 30 * s + 5))); Zc, _ = dense.consensus_codes_cl(Zs)
            l0 = int(round(float((Zc != 0).sum(1).mean())))
            arms = dict(consensus=Zc, single=Zs[0], **{c: dense.split_signed(dense.control_codes(c, E, 0, seed=3 * s)) for c in ("pca", "varimax", "ica")})
            arms["sparse_varimax"] = dense.topk_rows(arms["varimax"], l0)
            for arm, Z in arms.items():
                rows.append(dict(emb=tgt, seed=s, arm=arm, n_fine=len(fine), n_within_possible=sum(len(v) for v in subl.values() if len(v) >= 2),
                                 **scores(Z, lab, lin, fine), sec=time.time() - t0))
            pd.DataFrame(rows).to_csv(out, index=False)
            print(tgt, s, f"{time.time() - t0:.0f}s", {r["arm"]: (r["lin_cov"], r["fine_cov"], r["within_cov"]) for r in rows[-6:]}, flush=True)


def table(csv):
    df = pd.read_csv(csv); g = df.groupby(["emb", "arm"]); m = g[["lin_cov", "fine_cov", "within_cov"]].mean(); s = g[["lin_cov", "fine_cov", "within_cov"]].std(ddof=1).fillna(0)
    nf = df.groupby("emb")[["n_fine", "n_within_possible"]].first(); name = {"P:Geneformer": "Geneformer", "B:CADAST": "B, CadaST"}
    f = lambda e, a, c: f"{m.loc[(e, a), c]:.1f}" + (f"±{s.loc[(e, a), c]:.1f}" if s.loc[(e, a), c] > 0 else "")
    print("| Embedding | Fine types (subtypes) | Consensus | Single fit | Sparse varimax | Varimax | PCA | ICA |\n|---|---|---|---|---|---|---|---|")
    for e in [t for t in TARGETS if t in nf.index]:
        cells = [" / ".join(f(e, a, c) for c in ("lin_cov", "fine_cov", "within_cov")) for a in ("consensus", "single", "sparse_varimax", "varimax", "pca", "ica")]
        print(f"| {name.get(e, e.replace(':', ', '))} | {nf.loc[e, 'n_fine']} ({nf.loc[e, 'n_within_possible']}) | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if sys.argv[1] == "table": table(sys.argv[2] if len(sys.argv) > 2 else DATA / "results/FINE_TYPES.csv")
    else: run([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2], sys.argv[3].split(",") if len(sys.argv) > 3 else TARGETS)
