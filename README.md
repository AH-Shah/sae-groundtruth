# sae-groundtruth

Code for "Ground-truth evaluation of sparse autoencoder features on spatial, single-cell and language-model embeddings" (Shahsavand, Forghani et al., bioRxiv 2026).

## Install

```
pip install -r requirements.txt
```

## Data

Download our Zenodo deposit [DOI] into `data/`, then rebuild the public count data:

```
python scripts/build_inputs.py A                          # Crowell et al., section 232 (Zenodo 10.5281/zenodo.15574384)
python scripts/build_inputs.py B path/to/S0_exprMat_file  # Bruker CosMx Human WTx Colon Dataset
python scripts/build_inputs.py P path/to/pelka.h5ad       # Pelka et al. 2021 (GEO GSE178341)
python scripts/build_inputs.py hallmark                   # MSigDB Hallmark 2020
LLM_N=1000000 LLM_SPLIT=train-00000-of-00002 LLM_OUT=LLM_pythia70m_L3_1M.npz python scripts/embed_llm.py   # 1M-token activations
```

See `data/README.md` for details. `data/genes_2000.tsv` lists the gene panel.

## Reproduce

Tables and figures from the deposited results:

```
python scripts/make_tables.py
python scripts/make_figures.py
```

All experiments (about 25 GPU hours): `sh scripts/reproduce_all.sh`.

Single run, for example:

```
SAEGT_SEED=1 python scripts/run_embeddings.py B tsae_cons,tsae,varimax_topk Expr50,CellCharter,CADAST
```

`tests/test_equivalence.py` checks the code and data against the original analysis.

## License

MIT
