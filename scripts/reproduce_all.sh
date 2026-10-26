#!/bin/sh
# Rerun every experiment in the paper from the deposited inputs (data/inputs), as the research queues did (antig/N2/tools/78*.sh):
# ~25 GPU hours on an Apple M-series GPU. Outputs go to results/runs with the research file names; then build tables and figures with
#   SAEGT_RESULTS=results/runs python scripts/make_tables.py && python scripts/make_figures.py
# SAE fits on GPU are not bitwise deterministic: reruns reproduce the paper's numbers within the reported seed spread
# (rotations and everything on CPU reproduce exactly; see tests/test_equivalence.py).
set -e; cd "$(dirname "$0")/.."; PY=${PY:-python}
for s in 0 1 2; do
  export SAEGT_SEED=$s
  # spatial (Table 1, Figs. 1-2): main SAE (plain BatchTopK) + sparse varimax; rotations + Matryoshka (Table S1)
  $PY scripts/run_embeddings.py A tsae_cons,tsae,varimax_topk Expr50,CellCharter
  $PY scripts/run_embeddings.py B tsae_cons,tsae,varimax_topk Expr50,CellCharter,CADAST
  $PY scripts/run_embeddings.py A msae_cons,msae,pca,varimax,ica,varimax_topk
  $PY scripts/run_embeddings.py B msae_cons,msae,pca,varimax,ica,varimax_topk
  # Geneformer: SAE arms at 256 latents (dictionary-size rule); rotations (M-independent) with the default run
  SAEGT_M=256 $PY scripts/run_embeddings.py P tsae_cons,tsae,varimax_topk
  SAEGT_M=256 $PY scripts/run_embeddings.py P msae_cons,msae
  $PY scripts/run_embeddings.py P msae_cons,msae,pca,varimax,ica,varimax_topk
  # planted synthetic data (Fig. 3): Matryoshka + rotations, then plain BatchTopK + sparse varimax
  SAEGT_ARMS=SAEcons_Expr50,SAE_Expr50,PCA50,varimax50,varimax50_topk $PY scripts/run_synth.py
  SAEGT_ARMS=TSAEcons_Expr50,TSAE_Expr50,varimax50_topk SAEGT_TAG=_T $PY scripts/run_synth.py
  # language model, 1M tokens
  SAEGT_ARMS=tsae_cons,tsae,pca,varimax,ica,varimax_topk SAEGT_L0_FROM=tsae_cons $PY scripts/run_llm.py
  # two-stage residual model on planted data (Table S3)
  SAEGT_SETTINGS=20000:500 SAEGT_ARMS=TRES10_Expr50,TRES25_Expr50,TRES25s_Expr50,TRES25only_Expr50 SAEGT_TAG=_RES $PY scripts/run_synth.py
done
export SAEGT_SEED=0
# one-seed follow-ups: dictionary size on Geneformer (Results), compute-matched single fit (Table S2), residual stage on real data (Table S3)
SAEGT_M=1152 $PY scripts/run_embeddings.py P msae_cons,msae
SAEGT_STEPS=15000 $PY scripts/run_embeddings.py A msae Expr50,CellCharter
SAEGT_STEPS=15000 $PY scripts/run_embeddings.py B msae Expr50,CellCharter,CADAST
$PY scripts/run_embeddings.py B tsae_res10,tsae_res25,tsae_res25only CellCharter
$PY scripts/geometry.py; $PY scripts/check_l0.py
