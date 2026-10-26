"""Builds the paper's tables from the run results (data/results or SAEGT_RESULTS) -> results/.
"""
from pathlib import Path
import numpy as np, pandas as pd
import os
ROOT = Path(__file__).resolve().parents[1]; D = Path(os.environ.get("SAEGT_RESULTS", ROOT / "data/results")); O = ROOT / "results"; O.mkdir(exist_ok=True)
OLD = "msae_cons+msae+pca+varimax+ica+varimax_topk"; SEEDS = [(0, ""), (1, "_seed1"), (2, "_seed2")]
KEEP = {"A": ["Expr50", "CellCharter"], "B": ["Expr50", "CellCharter", "CADAST"], "P": ["Geneformer"]}
HI = dict(n_cov=1, R1_rep=1, R1_n=1, mono=1, n_mono=1, hallmark=1, knn=1, chimeric_ref=-1, rec_state_hi=1, rec_state_mid=1, rec_state_lo=1, id_dup=-1)
MAIN = ["tsae_cons", "tsae", "varimax", "varimax_topk", "ica", "pca", "msae_cons", "msae"]


def rd(f, s): d = pd.read_csv(f); d["seed"] = s; return d


def real():
    out = []
    for s, t in SEEDS:
        out += [rd(D / f"T2_A_unit_batch_rs0_tsae_cons+tsae+varimax_topk_Expr50+CellCharter{t}.csv", s), rd(D / f"T2_B_unit_batch_rs0_tsae_cons+tsae+varimax_topk_Expr50+CellCharter+CADAST{t}.csv", s),
                rd(D / f"T2_P_unit_batch_rs0_tsae_cons+tsae+varimax_topk_M256{t}.csv", s), rd(D / f"T2_P_unit_batch_rs0_msae_cons+msae_M256{t}.csv", s)]
        for k in "ABP":
            o = rd(D / f"T2_{k}_unit_batch_rs0_{OLD}{t}.csv", s); keep = ["pca", "varimax", "ica"] + (["msae_cons", "msae"] if k != "P" else []); out.append(o[o.method.isin(keep)])
    d = pd.concat(out); d = d[d.apply(lambda r: r.input in KEEP[r.slide], axis=1)]
    big = sorted(x for x in D.glob("T2_LLMbig_*.csv") if "thr0bug" not in x.name); llm = big or sorted(D.glob("T2_LLM_*.csv"))
    for f in llm:
        g = pd.read_csv(f); g["slide"] = "L"; g = g[g.method != "lpsae"]
        if not big: g = g[g.method != "varimax_topk"] if "tsae_cons" not in set(g.method) else g
        d = pd.concat([d, g])
    d = d.drop_duplicates(["slide", "input", "method", "seed"], keep="last"); d["R1_n"] = d.R1_rep * d.alive; d["emb"] = d.slide + ":" + d.input
    cv = coverage(); d["n_cov"] = [cv.get((r.slide, r.input, r.method, r.seed), r.n_cls if r.slide == "L" else np.nan) for r in d.itertuples()]
    return d, bool(big)


def coverage():
    """distinct concepts covered: lineages (of 7) with >= 1 feature at selectivity >= .7, from the per-feature files of the same runs
    real() reads; LLM uses n_cls (token classes of 8); synthetic uses distinct planted programs matched (of 28) in synth()."""
    cv = {}
    for s, t in SEEDS:
        files = [(D / f"T2_A_unit_batch_rs0_tsae_cons+tsae+varimax_topk_Expr50+CellCharter{t}_features.csv", None), (D / f"T2_B_unit_batch_rs0_tsae_cons+tsae+varimax_topk_Expr50+CellCharter+CADAST{t}_features.csv", None),
                 (D / f"T2_P_unit_batch_rs0_tsae_cons+tsae+varimax_topk_M256{t}_features.csv", None), (D / f"T2_P_unit_batch_rs0_msae_cons+msae_M256{t}_features.csv", None)]
        files += [(D / f"T2_{k}_unit_batch_rs0_{OLD}{t}_features.csv", ["pca", "varimax", "ica"] + (["msae_cons", "msae"] if k != "P" else [])) for k in "ABP"]
        for fp, keep in files:
            d = pd.read_csv(fp)
            if keep: d = d[d.method.isin(keep)]
            for (sl, i, m), g in d.groupby(["slide", "input", "method"]):
                cv.setdefault((sl, i, m, s), g.loc[g.pmax >= .7, "lineage"].nunique())
    return cv


def synth():
    out = []
    for s, t in SEEDS:
        out.append(rd(D / f"SYNTH_ROT_T{t}.csv", s)); o = rd(D / f"SYNTH_ROT{t}.csv", s); out.append(o[o.method.isin(["PCA50", "varimax50", "SAEcons_Expr50", "SAE_Expr50"])])
    d = pd.concat(out); d["n_mono"] = d.get("n_mono", (d.mono * d.n_feat).round()).fillna((d.mono * d.n_feat).round())
    d["method"] = d.method.map(dict(TSAEcons_Expr50="tsae_cons", TSAE_Expr50="tsae", varimax50_topk="varimax_topk", PCA50="pca", varimax50="varimax", SAEcons_Expr50="msae_cons", SAE_Expr50="msae"))
    d["emb"] = "n" + (d.n // 1000).astype(str) + "k_d" + d.depth.astype(str); d["R1_n"] = d.R1_rep * d.n_feat
    d["n_cov"] = (d.rec_identity * 6 + (d.rec_state_hi + d.rec_state_mid + d.rec_state_lo) * 6 + d.rec_shared * 4).round()  # distinct programs matched (of 28)
    return d


def ms(g, m):
    t = g.groupby(["emb", "method"])[m].agg(["mean", "std"]); t["std"] = t["std"].fillna(0); return t


def cmp(d, a, b, m):
    if m not in d or d[m].isna().all(): return None
    t = ms(d, m); r = []
    for e in d.emb.unique():
        if (e, a) not in t.index or (e, b) not in t.index: continue
        x, y = t.loc[(e, a)], t.loc[(e, b)]
        if np.isnan(x["mean"]) or np.isnan(y["mean"]): continue
        dl = (x["mean"] - y["mean"]) * HI[m]; tol = x["std"] + y["std"] + 1e-9; r.append((1 if dl > tol else -1 if -dl > tol else 0, x["mean"] - y["mean"]))
    if not r: return None
    v = np.array([z[0] for z in r]); return f"{(v == 1).sum()}/{(v == 0).sum()}/{(v == -1).sum()} ({np.median([z[1] for z in r]):+.2f})"


def fmt(t, e, a, p):
    if (e, a) not in t.index or np.isnan(t.loc[(e, a), "mean"]): return "-"
    return f"{t.loc[(e, a), 'mean']:.{p}f}±{t.loc[(e, a), 'std']:.{p}f}"


def main():
    R, big = real(); S = synth(); R.to_csv(O / "REAL_v3.csv", index=False); S.to_csv(O / "SYNTH_v3.csv", index=False)
    src = dict(slides=R[R.slide.isin(["A", "B"])], synth=S, geneformer=R[R.slide == "P"], llm=R[R.slide == "L"])
    mets = ["n_cov", "n_mono", "mono", "chimeric_ref", "hallmark", "knn", "R1_rep", "R1_n", "rec_state_hi", "rec_state_mid", "rec_state_lo", "id_dup"]
    L = [f"# SAE paper tables v3 (main SAE = plain BatchTopK; LLM leg = {'1M tokens' if big else '50k tokens'})", "",
         "## consensus (tsae_cons) vs each method: W/T/L beyond seed sd (median diff)", "| source | vs | " + " | ".join(mets) + " |", "|---|---|" + "---|" * len(mets)]
    for k, d in src.items():
        for b in ["tsae", "varimax", "varimax_topk", "ica", "pca", "msae_cons", "msae"]:
            L.append(f"| {k} | {b} | " + " | ".join(cmp(d, "tsae_cons", b, m) or "-" for m in mets) + " |")
    L += ["", "## per embedding (mean±sd)", "| emb | metric | " + " | ".join(MAIN) + " |", "|---|---|" + "---|" * len(MAIN)]
    for m in ["n_cov", "R1_rep", "R1_n", "alive", "L0", "n_mono", "mono", "chimeric_ref", "hallmark", "knn", "sec"]:
        t = ms(R, m)
        for e in R.emb.unique(): L.append(f"| {e} | {m} | " + " | ".join(fmt(t, e, a, 0 if m in ("R1_n", "alive", "n_mono", "sec") else 2) for a in MAIN) + " |")
    L += ["", "## synthetic (mean±sd)", "| setting | arm | mono | n_mono | R1 | rec hi | rec mid | rec lo | id_dup | n_feat |", "|---|---|---|---|---|---|---|---|---|---|"]
    for e in S.emb.unique():
        for a in ["tsae_cons", "tsae", "pca", "varimax", "varimax_topk", "msae_cons", "msae"]:
            L.append(f"| {e} | {a} | " + " | ".join(fmt(ms(S, m), e, a, 0 if m in ("n_mono", "n_feat") else 2) for m in ["mono", "n_mono", "R1_rep", "rec_state_hi", "rec_state_mid", "rec_state_lo", "id_dup", "n_feat"]) + " |")
    (O / "TABLES_v3.md").write_text("\n".join(L) + "\n"); print("wrote TABLES_v3.md", len(L))


if __name__ == "__main__":
    main()
