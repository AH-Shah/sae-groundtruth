"""Planted synthetic data: SAEs vs rotations on an Expr50 embedding with known programs (recovery by state strength, programs matched,
reproducibility). Adapted from antig/N2/tools/55b_dsae_synth_lp.py (run_rot); LaPrune arms removed, output path configurable.

usage: python scripts/run_synth.py
  env: SAEGT_SEED (0, 1, 2; also the simulation seed), SAEGT_ARMS (default = the paper's main arms), SAEGT_SETTINGS ("20000:150,20000:500,60000:150"),
       SAEGT_TAG (output suffix), SAEGT_OUT (results/runs)
  arms: TSAEcons_Expr50, TSAE_Expr50, varimax50_topk (needs a consensus arm earlier), PCA50, varimax50, SAEcons_Expr50, SAE_Expr50 (Matryoshka),
        TRES<pct>[s|only]_Expr50 (two-stage residual model)
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from saegt.synth import simulate, log1p, prep, score, mono_share  # noqa: E402
OUT = Path(os.environ.get("SAEGT_OUT", ROOT / "results/runs")); OUT.mkdir(parents=True, exist_ok=True); SYN_SEED = int(os.environ.get("SAEGT_SEED", "0"))


def run_rot():
    import torch
    from sklearn.decomposition import PCA
    from saegt import sae as dense, residual
    torch.set_num_threads(8); mps = "mps" if torch.backends.mps.is_available() else "cpu"; rows = []
    for n, depth in [tuple(int(v) for v in x.split(":")) for x in os.environ.get("SAEGT_SETTINGS", "20000:150,20000:500,60000:150").split(",")]:
        t0 = time.time(); C, U, kind, lin_of, lin, pat = simulate(n, depth, seed=SYN_SEED); L = log1p(C)
        X = prep(PCA(50, random_state=0).fit_transform(L - L.mean(0))); half = np.random.default_rng(1 + 100 * SYN_SEED).permutation(n) < n // 2
        cfg = dict(M=200, k=8, steps=3000, device=mps); l0 = {}

        def fit(arm, Xf, s):
            s = s + 3 * SYN_SEED
            if arm == "PCA50": return dense.split_signed(dense.control_codes("pca", X, 0, seed=s, Xfit=Xf))
            if arm == "varimax50": return dense.split_signed(dense.control_codes("varimax", X, 0, seed=s, Xfit=Xf))
            if arm == "varimax50_topk": return dense.topk_rows(dense.split_signed(dense.control_codes("varimax", X, 0, seed=s, Xfit=Xf)), l0["k"])
            if arm == "SAE_Expr50": return dense.encode(dense.fit(Xf, {**cfg, "seed": s})[0], X)
            if arm.startswith("TRES"):
                tag = arm[4:].split("_")[0]; frac = int("".join(ch for ch in tag if ch.isdigit())) / 100
                Zc, Z2, _ = residual.two_stage(Xf, X, {**cfg, "matryoshka": False}, frac=frac, stage2="single" if tag.endswith("s") else "cons", seeds=tuple(range(10 * s, 10 * s + 5)))
                return Z2 if tag.endswith("only") else Zc
            if arm == "TSAE_Expr50": return dense.encode(dense.fit(Xf, {**cfg, "seed": s, "matryoshka": False})[0], X)
            if arm == "TSAEcons_Expr50": return dense.consensus(Xf, X, {**cfg, "matryoshka": False}, seeds=tuple(range(10 * s, 10 * s + 5)))[0]
            return dense.consensus(Xf, X, cfg, seeds=tuple(range(10 * s, 10 * s + 5)))[0]  # SAEcons_Expr50 (Matryoshka consensus)
        for arm in os.environ.get("SAEGT_ARMS", "TSAEcons_Expr50,TSAE_Expr50,varimax50_topk,PCA50,varimax50").split(","):
            Z = fit(arm, X, 0)
            if arm in ("SAEcons_Expr50", "TSAEcons_Expr50"): l0["k"] = int(round(float((Z != 0).sum(1).mean())))
            rep, rmed = dense.act_match(fit(arm, X[half], 1), fit(arm, X[~half], 2))
            r = score(Z, U, kind, lin_of, lin); r.update(method=arm, n=n, depth=depth, seed=SYN_SEED, n_mono=int(round(mono_share(Z, U) * dense.alive(Z).sum())), mono=mono_share(Z, U), R1_rep=rep, R1_rmed=rmed, L0=float((Z != 0).sum(1).mean()))
            rows.append(r); pd.DataFrame(rows).to_csv(OUT / f"SYNTH_ROT{os.environ.get('SAEGT_TAG', '')}{f'_seed{SYN_SEED}' if SYN_SEED else ''}.csv", index=False)
            print(f"n{n} d{depth} {arm}: R1 {rep:.2f} mono {r['mono']:.2f} L0 {r['L0']:.1f} st {r['rec_state_hi']:.2f}/{r['rec_state_mid']:.2f}/{r['rec_state_lo']:.2f} dup {r['id_dup']:.1f}", flush=True)
        print(f"setting {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run_rot()
