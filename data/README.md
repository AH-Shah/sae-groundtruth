# Data

Our Zenodo deposit [DOI] goes into `data/inputs/` and `data/results/`: embeddings, cell IDs and cell types of the analyzed cells,
language-model activations, reference markers, count checksums and all run results.

Count data and gene sets are rebuilt from their sources into `data/built/` with `scripts/build_inputs.py`; the 1M-token language-model file is rebuilt into `data/inputs/`:

| Data | Source |
|---|---|
| Section A | Crowell et al., section 232, `232.h5ad`, Zenodo 10.5281/zenodo.15574384 |
| Section B | Bruker Spatial Biology, CosMx Human Whole Transcriptome Colon Dataset, `S0_exprMat_file` |
| Pelka atlas | Pelka et al. 2021, GEO GSE178341 (CELLxGENE h5ad) |
| Hallmark | MSigDB Hallmark 2020 via Enrichr |
| Language model, 1M tokens | `LLM_N=1000000 LLM_SPLIT=train-00000-of-00002 LLM_OUT=LLM_pythia70m_L3_1M.npz python scripts/embed_llm.py` (WikiText-103 + Pythia-70m; rebuilds the file used in the paper exactly) |

`genes_2000.tsv`: the 2,000-gene panel; housekeeping genes (Eisenberg & Levanon 2013) are flagged.
