# Data

Our Zenodo deposit [DOI] goes into `data/inputs/` and `data/results/`: embeddings, cell IDs and cell types of the analyzed cells,
language-model activations, reference markers, count checksums and all run results.

Count data and gene sets are rebuilt from their sources into `data/built/` with `scripts/build_inputs.py`:

| Data | Source |
|---|---|
| Section A | Crowell et al., section 232, `232.h5ad`, Zenodo 10.5281/zenodo.15574384 |
| Section B | Bruker Spatial Biology, CosMx Human Whole Transcriptome Colon Dataset, `S0_exprMat_file` |
| Pelka atlas | Pelka et al. 2021, GEO GSE178341 (CELLxGENE h5ad) |
| Hallmark | MSigDB Hallmark 2020 via Enrichr |

`genes_2000.tsv`: the 2,000-gene panel; housekeeping genes (Eisenberg & Levanon 2013) are flagged.
