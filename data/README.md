# Data

Download the Zenodo deposit [DOI] and place (or symlink) its two folders here: `data/inputs/` and `data/results/`. Verify with
`shasum -a 256 -c SHA256SUMS_inputs.txt` from inside the deposit.

| File (`data/inputs/`) | Content | Size |
|---|---|---|
| `section_A.npz` | CosMx WTx colorectal section A (Crowell et al.), the 50,000 cells analyzed: counts (CSR, 2,000 genes), genes, barcodes, cell types (InSituType, confidence >= 0.7, else empty), embeddings `emb_Expr50`, `emb_CellCharter` | 65 MB |
| `section_B.npz` | CosMx WTx Human Colon Dataset (Bruker), the 50,000 cells analyzed: as above plus `emb_CADAST` | 121 MB |
| `pelka_geneformer.npz` | 12,881 Pelka et al. cells: Geneformer V2 embedding `emb_Geneformer`, counts on the CosMx panel genes, cell types | 63 MB |
| `LLM_pythia70m_L3_1M.npz` | Pythia-70m layer-3 residual stream, 1M WikiText-103 training tokens (float16), token strings, token classes | 1.19 GB |
| `LLM_pythia70m_L3.npz` | the same for 50k validation tokens (preliminary analysis, Supplementary Table S4) | 60 MB |
| `markers.json` | reference lineage markers (seed/eval sets) derived from Pelka et al. | 3 kB |
| `hallmark.json` | MSigDB Hallmark 2020 gene sets (Enrichr library) | 66 kB |

`data/results/` holds the per-run CSVs that produced every number in the paper (`T2_*.csv`: spatial, Geneformer and language-model runs,
one file per seed; `SYNTH_ROT*.csv`: planted data), their per-feature tables (`*_features.csv`) and the final tables `TABLES_v3.md`.

`data/genes_2000.tsv` (in the repository): the 2,000 genes used for both sections, in matrix column order. The column
`housekeeping_eisenberg_levanon_2013` flags the 207 genes on the Eisenberg & Levanon (2013) human housekeeping list (see Methods,
"Gene panel", for how they entered the panel).
