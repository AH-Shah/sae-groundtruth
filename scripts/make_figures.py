"""[Ported from antig/N2/tools/94_sae_figs_v3.py by tools/97_port_scripts.py: paths (run make_tables.py first)]
SAE paper figures v3 (main SAE = plain BatchTopK; rows 337-340) from trials/sae_paper/{REAL,SYNTH}_v3.csv -> paper/figs/fig{1,2,3}_v3.{pdf,png}.
Palette validated (dataviz validator, light mode: CVD/normal-vision PASS; low-contrast hues carry marker shape + legend)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; F = str(ROOT / "figures") + "/"; Path(F).mkdir(exist_ok=True)
R = pd.read_csv(ROOT / "results/REAL_v3.csv"); S = pd.read_csv(ROOT / "results/SYNTH_v3.csv")
ARM = dict(tsae_cons=("consensus SAE", "#2a78d6", "o"), tsae=("single SAE", "#eb6834", "s"), pca=("PCA", "#1baf7a", "^"),
           varimax=("varimax", "#eda100", "D"), varimax_topk=("varimax @ SAE L0", "#e87ba4", "v"), ica=("ICA", "#008300", "P"))
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": .5, "lines.linewidth": 2})
agg = lambda d, keys, cols: d.groupby(keys)[cols].agg(["mean", "std"]).fillna(0)


def legend(fig, ax, n, y=0):
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=n, frameon=False, fontsize=7, bbox_to_anchor=(.5, y))


def fig1():
    """x = R1, y = distinct concepts covered (row 341); faint lines join single and consensus SAE of the same embedding."""
    fig, ax = plt.subplots(1, 4, figsize=(9.6, 2.8))
    srcs = [("Spatial (5 embeddings)", R[R.slide.isin(["A", "B"])], "lineages covered (of 7)", 7),
            ("Planted synthetic (3 settings)", S, "planted programs matched (of 28)", 28),
            ("Geneformer V2 (scRNA)", R[R.slide == "P"], "lineages covered (of 7)", 7),
            ("Pythia-70m (text, 1M tokens)", R[R.slide == "L"], "token classes covered (of 8)", 8)]
    for a, (t, d, yl, top) in zip(ax, srcs):
        g = agg(d, ["emb", "method"], ["R1_rep", "n_cov"])
        for e in d.emb.unique():
            if (e, "tsae") in g.index and (e, "tsae_cons") in g.index:
                a.plot([g.loc[(e, "tsae"), ("R1_rep", "mean")], g.loc[(e, "tsae_cons"), ("R1_rep", "mean")]],
                       [g.loc[(e, "tsae"), ("n_cov", "mean")], g.loc[(e, "tsae_cons"), ("n_cov", "mean")]], color="#c9c8c3", lw=.8, zorder=1)
        for m, (lab, c, mk) in ARM.items():
            if m not in g.index.get_level_values(1): continue
            x = g.xs(m, level=1)
            a.errorbar(x[("R1_rep", "mean")], x[("n_cov", "mean")], xerr=x[("R1_rep", "std")], yerr=x[("n_cov", "std")], fmt=mk, ms=5, color=c,
                       mec="white", mew=.6, elinewidth=.8, ls="none", label=lab, zorder=3)
        a.set_xlim(-.03, 1.05); a.set_ylim(-.04 * top, 1.06 * top); a.set_title(t, fontsize=8, color=INK, loc="left"); a.set_xlabel("R1 reproducible share"); a.set_ylabel(yl)
    legend(fig, ax[0], 6); fig.tight_layout(rect=(0, .09, 1, 1)); fig.savefig(F + "fig1_v3.pdf"); fig.savefig(F + "fig1_v3.png", dpi=200)


def fig2():
    d = R[R.slide.isin(["A", "B", "P", "L"]) & R.method.isin(["tsae", "tsae_cons"])]; g = agg(d, ["emb", "method"], ["R1_rep", "R1_n", "n_mono"]); E = list(dict.fromkeys(d.emb))
    lab_e = [{"P:Geneformer": "Geneformer", "L:Pythia70m_L3_1M": "Pythia-70m"}.get(e, e.replace("CADAST", "CadaST")) for e in E]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7)); y = np.arange(len(E))
    for j, (col, ttl, xl, log) in enumerate([("R1_rep", "reproducible share", "R1 share (random halves)", False), ("R1_n", "reproducible features", "R1 x features (log)", True),
                                            ("n_mono", "concept-specific features", "features with selectivity >= .7 (log)", True)]):
        a = ax[j]
        for m, dy in [("tsae", -.15), ("tsae_cons", .15)]:
            lab, c, mk = ARM[m]; x = g.xs(m, level=1).loc[E]
            a.errorbar(x[(col, "mean")] + (1 if log else 0), y + dy, xerr=x[(col, "std")], fmt=mk, color=c, ms=5, mec="white", mew=.6, elinewidth=.8, ls="none", label=lab)
        a.set_yticks(y); a.set_yticklabels(lab_e if j == 0 else []); a.set_title(ttl, fontsize=8, loc="left", color=INK); a.grid(axis="y", visible=False); a.set_xlabel(xl)
        if log: a.set_xscale("log"); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    legend(fig, ax[0], 2); fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(F + "fig2_v3.pdf"); fig.savefig(F + "fig2_v3.png", dpi=200)


def fig3():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True); x = np.arange(3); arms = ["tsae_cons", "tsae", "pca", "varimax", "varimax_topk"]
    for a, e in zip(ax, ["n20k_d150", "n60k_d150", "n20k_d500"]):
        g = agg(S[S.emb == e], ["method"], ["rec_state_hi", "rec_state_mid", "rec_state_lo"])
        for i, m in enumerate(arms):
            lab, c, mk = ARM[m]
            mu = [g.loc[m, (f"rec_state_{s}", "mean")] for s in ("hi", "mid", "lo")]; sd = [g.loc[m, (f"rec_state_{s}", "std")] for s in ("hi", "mid", "lo")]
            a.errorbar(x + (i - 2) * .04, mu, yerr=sd, marker=mk, color=c, ms=4, mec="white", mew=.5, elinewidth=.8, lw=1.5, label=lab)
        a.set_xticks(x); a.set_xticklabels(["strong", "mid", "weak"]); a.set_title(f"{e[1:3]}k cells, depth {e.split('_d')[1]}", fontsize=8, loc="left", color=INK)
        a.set_ylabel("states recovered"); a.yaxis.set_tick_params(labelleft=True); a.set_xlabel("planted state strength")
    ax[0].set_ylim(-.05, 1.05); legend(fig, ax[0], 5); fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(F + "fig3_v3.pdf"); fig.savefig(F + "fig3_v3.png", dpi=200)


fig1(); fig2(); fig3(); print("figs v3 ok")
