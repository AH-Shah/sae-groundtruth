"""Checks the repository against the original analysis code (SAEGT_RESEARCH=path) and the deposited results.
usage: python tests/test_equivalence.py [--quick]
"""
import importlib.util, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); RES = os.environ.get("SAEGT_RESEARCH"); QUICK = "--quick" in sys.argv
from saegt import metrics as mt, residual as r_new, sae as s_new, synth as y_new  # noqa: E402
OK = []


def check(name, ok, detail=""):
    OK.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def same(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray): return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')
    if isinstance(a, float): return (np.isnan(a) and np.isnan(b)) or a == b
    return a == b


def research():
    sys.path.insert(0, RES)
    from rebuild.sae import dense as s_old, residual as r_old, evaluate as ev
    sp = importlib.util.spec_from_file_location("s55", Path(RES) / "tools/55_dsae_synth.py"); y_old = importlib.util.module_from_spec(sp); sp.loader.exec_module(y_old)
    sp = importlib.util.spec_from_file_location("t53b", Path(RES) / "tools/53b_dsae_slide_flags.py"); t = importlib.util.module_from_spec(sp); sp.loader.exec_module(t)
    sp = importlib.util.spec_from_file_location("bench", Path(RES) / "bench/bench.py"); b = importlib.util.module_from_spec(sp); sp.loader.exec_module(b)
    return s_old, r_old, ev, y_old, t, b


def level1(s_old, r_old, ev, y_old, t, b):
    import torch
    torch.use_deterministic_algorithms(False); g = np.random.default_rng(0)
    X = (np.abs(g.normal(size=(3000, 12))) @ g.normal(size=(12, 24)) + .3 * g.normal(size=(3000, 24))).astype(np.float32); X = y_new.prep(X)
    for mat in (True, False):
        cfg = dict(M=48, k=4, steps=60, device="cpu", seed=3, matryoshka=mat)
        torch.manual_seed(0); mo, ho = s_old.fit(X, cfg); torch.manual_seed(0); mn, hn = s_new.fit(X, cfg)
        check(f"sae.fit + encode (matryoshka={mat})", same(s_old.encode(mo, X), s_new.encode(mn, X)) and same(ho, hn))
    cfg = dict(M=48, k=4, steps=40, device="cpu", matryoshka=False)
    check("sae.consensus", same(s_old.consensus(X[:2000], X, cfg), s_new.consensus(X[:2000], X, cfg)))
    for kind in ("pca", "varimax", "ica"):
        check(f"sae.control_codes({kind}) + split_signed", same(s_old.split_signed(s_old.control_codes(kind, X, 0, seed=1, Xfit=X[:2000])), s_new.split_signed(s_new.control_codes(kind, X, 0, seed=1, Xfit=X[:2000]))))
    Z = np.maximum(X @ g.normal(size=(24, 40)).astype(np.float32), 0); Z2 = np.maximum(X @ g.normal(size=(24, 40)).astype(np.float32), 0)
    check("sae.topk_rows / alive / act_match", same(s_old.topk_rows(Z, 5), s_new.topk_rows(Z, 5)) and same(s_old.alive(Z), s_new.alive(Z)) and same(s_old.act_match(Z, Z2), s_new.act_match(Z, Z2)))
    lin = np.array(mt.LINS)[g.integers(0, 7, len(Z))]; genes = np.array([f"G{i}" for i in range(300)]); L = np.abs(g.normal(size=(len(Z), 300))).astype(np.float32)
    check("sae.lineage_selectivity / gene_programs", same(s_old.lineage_selectivity(Z, lin, mt.LINS), s_new.lineage_selectivity(Z, lin, mt.LINS)) and same(s_old.gene_programs(Z, L, genes), s_new.gene_programs(Z, L, genes)))
    cfg = dict(M=40, k=4, steps=30, device="cpu", matryoshka=False)
    r_old._CACHE.clear(); r_new._CACHE.clear()
    check("residual.two_stage", same(r_old.two_stage(X[:2000], X, cfg, frac=.25, stage2="cons", seeds=(0, 1, 2)), r_new.two_stage(X[:2000], X, cfg, frac=.25, stage2="cons", seeds=(0, 1, 2))))
    hm = ev.f0.hallmark(); gs = np.array(sorted({x for v in hm.values() for x in v})[:2000]); tops = [list(g.choice(gs, 50, replace=False)) for _ in range(20)]
    check("metrics.enrich (Hallmark)", all(same(ev.f0.enrich(tp, gs, hm), mt.enrich(tp, gs, hm)) for tp in tops) and same(hm, mt.hallmark(ROOT / "data/built/hallmark.json")))
    import json; mk = json.load(open(Path(RES) / "trials/markers/markers.json"))["cosmx2000"]; mkg = sorted({x for v in mk.values() for x in v["seed"] + v["eval"]})
    tops = [list(g.choice(mkg, 50, replace=False)) for _ in range(50)]; ls = list(g.choice(mt.LINS, 50))
    check("metrics.ref_foreign / foreign_markers", all(t.ref_foreign(tp, l, mk) == mt.ref_foreign(tp, l, mk) for tp, l in zip(tops, ls)) and all(ev.f0.foreign_markers(";".join(tp), l) == mt.foreign_markers(";".join(tp), l) for tp, l in zip(tops, ls)))
    lab = np.array(["EpiT", "Fibro", "Macro", "TCD8", "Plasma", ""])[g.integers(0, 6, len(Z))]; ii = lab != ""
    check("metrics.bench_lineage (kNN)", same(ev.bench_lineage(Z[ii], lab[ii]), mt.bench_lineage(Z[ii], lab[ii])) and same((t.LMAP, t.LINS, t.REFM), (mt.LMAP, mt.LINS, mt.REFM)))
    C1 = y_old.simulate(3000, 150, seed=1); C2 = y_new.simulate(3000, 150, seed=1); check("synth.simulate", same(C1, C2))
    C, U, kind, lin_of, lin, pat = C2; Zs = np.maximum(y_new.prep(y_new.log1p(C)[:, :60]) @ g.normal(size=(60, 30)).astype(np.float32), 0)
    check("synth.score / mono_share / log1p / prep", same(y_old.score(Zs, U, kind, lin_of, lin), y_new.score(Zs, U, kind, lin_of, lin)) and same(y_old.mono_share(Zs, U), y_new.mono_share(Zs, U)))


def level2(t, ev):
    import anndata as ad, scipy.sparse as sp
    for key in ("A", "B"):
        c = t.SL[key]; a = ad.read_h5ad(c["h5"], backed="r"); bc = a.obs_names.astype(str).values; rows = np.sort(np.random.default_rng(0).choice(len(bc), 50000, replace=False))
        z = np.load(ROOT / f"data/inputs/section_{key}.npz", allow_pickle=True); cz = np.load(ROOT / f"data/built/counts_{key}.npz"); C = sp.csr_matrix((cz["data"], cz["indices"], cz["indptr"]), shape=tuple(cz["shape"]))
        ok = (C != sp.csr_matrix(a.X[rows]).astype(np.float32)).nnz == 0 and same(z["celltype"].astype(str), ev.labels(c["slide"], bc)[rows]) and same(z["genes"].astype(str), np.asarray(a.var_names).astype(str))
        for name in z["embeddings"]: ok = ok and same(z[f"emb_{name}"], t.read_latent(c["inputs"][name], bc)[rows])
        check(f"deposited section_{key} + rebuilt counts == research readers (counts, labels, genes, {', '.join(z['embeddings'])})", ok)
    E, C, genes, mid, bc = t.pelka_inputs(); z = np.load(ROOT / "data/inputs/pelka_geneformer.npz", allow_pickle=True); c = np.load(ROOT / "data/built/counts_P.npz")
    check("deposited pelka_geneformer + rebuilt counts == research reader", same(z["emb_Geneformer"], E) and (sp.csr_matrix((c["data"], c["indices"], c["indptr"]), shape=tuple(c["shape"])) != C).nnz == 0 and same(z["celltype"].astype(str), mid) and same(z["genes"].astype(str), genes))


def level3():
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "SAEGT_OUT": tmp, "SAEGT_SEED": "0"}
        subprocess.run([sys.executable, str(ROOT / "scripts/run_embeddings.py"), "B", "pca,varimax,ica", "Expr50"], env=env, check=True, capture_output=True)
        new = pd.read_csv(Path(tmp) / "T2_B_unit_batch_rs0_pca+varimax+ica_Expr50.csv")
        old = pd.read_csv(ROOT / "data/results/T2_B_unit_batch_rs0_msae_cons+msae+pca+varimax+ica+varimax_topk.csv")
        cols = ["alive", "R1_rep", "R1_rmed", "knn", "purity", "mono", "n_mono", "hallmark", "chimeric", "chimeric_ref"]
        for m in ("pca", "varimax", "ica"):
            a = new[new.method == m].iloc[0]; b = old[(old.input == "Expr50") & (old.method == m)].iloc[0]
            bad = [c for c in cols if not np.isclose(a[c], b[c], rtol=0, atol=1e-12)]
            check(f"end-to-end B Expr50 {m}: rerun == paper run CSV", not bad, f"({len(cols)} metrics; differing: {bad})" if bad else f"({len(cols)} metrics)")
        if not QUICK:
            env2 = {**env, "SAEGT_SETTINGS": "20000:150", "SAEGT_ARMS": "PCA50,varimax50"}
            subprocess.run([sys.executable, str(ROOT / "scripts/run_synth.py")], env=env2, check=True, capture_output=True)
            new = pd.read_csv(Path(tmp) / "SYNTH_ROT.csv"); old = pd.read_csv(ROOT / "data/results/SYNTH_ROT.csv")
            cols = ["rec_identity", "rec_state_hi", "rec_state_mid", "rec_state_lo", "rec_shared", "id_dup", "n_feat", "mono", "R1_rep"]
            for m in ("PCA50", "varimax50"):
                a = new[new.method == m].iloc[0]; b = old[(old.method == m) & (old.depth == 150) & (old.n == 20000)].iloc[0]
                bad = [c for c in cols if not np.isclose(a[c], b[c], rtol=0, atol=1e-12)]
                check(f"end-to-end synthetic n20k d150 {m}: rerun == paper run CSV", not bad, f"differing: {bad}" if bad else f"({len(cols)} metrics)")


def level4():
    subprocess.run([sys.executable, str(ROOT / "scripts/make_tables.py")], check=True, capture_output=True)
    new = (ROOT / "results/TABLES_v3.md").read_text(); old = (ROOT / "data/results/TABLES_v3.md").read_text()
    check("make_tables.py regenerates the paper tables byte for byte", new == old, f"({len(new.splitlines())} lines)")


if __name__ == "__main__":
    if RES:
        o = research(); level1(*o); level2(o[4], o[2])
    else: print("SKIP  levels 1-2 (set SAEGT_RESEARCH to the research code directory)")
    level3(); level4()
    print(f"\n{sum(OK)}/{len(OK)} checks passed"); sys.exit(0 if all(OK) else 1)
