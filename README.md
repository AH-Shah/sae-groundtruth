# sae-groundtruth

Code and data for **"Ground-truth evaluation of sparse autoencoder features on spatial, single-cell and language-model embeddings"** (bioRxiv, link to be added).

We compare sparse autoencoders (SAEs), single fits and a five-seed consensus, with rotations of the same embedding (PCA, varimax, ICA and sparse varimax). The tests use embeddings whose concepts are known: CosMx spatial transcriptomics (cell lineages), planted synthetic data (programs), Geneformer embeddings of single-cell RNA (lineages) and the Pythia-70m residual stream (token classes). Each method is scored on reproducibility between random halves, concept coverage, gene-program purity, kNN identity and recovery of planted states.

## Contents

| Path | What it is |
|---|---|
| `saegt/sae.py` | BatchTopK and Matryoshka BatchTopK SAE, consensus SAE, rotation baselines, matching and selectivity metrics |
| `saegt/residual.py` | two-stage residual model (Results, "What does not remove the cost") |
| `saegt/metrics.py` | kNN identity, Hallmark enrichment, chimeric gene programs (reference and hand markers), lineage map |
| `saegt/synth.py` | planted synthetic data generator and ground-truth scores |
| `scripts/run_embeddings.py` | spatial sections A, B and Geneformer (P): all SAE and rotation arms |
| `scripts/run_synth.py` | planted synthetic data |
| `scripts/run_llm.py` | Pythia-70m, 1M tokens |
| `scripts/embed_llm.py`, `scripts/embed_geneformer.py` | regenerate the language-model and Geneformer embeddings (optional; deposited) |
| `scripts/make_tables.py`, `scripts/make_figures.py` | Table 1, supplementary numbers, Figs. 1-3 |
| `scripts/geometry.py`, `scripts/check_l0.py` | Supplementary Table S4; the GPU top-k threshold check (Methods) |
| `data/genes_2000.tsv` | the 2,000 genes of the spatial panel, housekeeping genes flagged (Methods, Gene panel) |
| `scripts/reproduce_all.sh` | every experiment in the paper, with the seeds and settings used |
| `tests/test_equivalence.py` | proof that this code is the code that produced the paper (see below) |

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

The paper used Python 3.11.15 on an Apple M-series GPU (PyTorch MPS). The code falls back to the CPU when no MPS device is available.

## Data

Download the Zenodo deposit [DOI] into `data/` (see `data/README.md`). It contains the exact inputs analyzed:
- for each CosMx section, the 50,000 cells used, with counts, cell types and the Expr50, CellCharter and CadaST embeddings;
- the Geneformer embeddings with counts for the same cells;
- both language-model token sets;
- marker and Hallmark gene sets;
- every per-run result file behind the paper's numbers.

The raw sources are:
- section A, from Crowell et al., "Tracing colorectal malignancy transformation from cell to tissue scale" [accession];
- section B, the Bruker CosMx Human Whole Transcriptome Colon Dataset (Khafizov et al., bioRxiv 10.1101/2024.11.27.625536);
- the Pelka et al. (2021) colorectal cancer atlas;
- WikiText-103 and Pythia-70m (Hugging Face).

## Reproduce the paper

**Tables and figures from the deposited run results (seconds):**

```
python scripts/make_tables.py      # -> results/TABLES_v3.md, REAL_v3.csv, SYNTH_v3.csv
python scripts/make_figures.py     # -> figures/fig{1,2,3}_v3.{pdf,png}
python scripts/geometry.py         # Supplementary Table S4
```

**All experiments from the deposited inputs (~25 GPU hours):** `sh scripts/reproduce_all.sh`. Then rebuild the tables from your runs with `SAEGT_RESULTS=results/runs python scripts/make_tables.py`. SAE training on a GPU is not bitwise deterministic, so reruns reproduce the paper's numbers within the reported seed spread. Rotations, and all computations on the CPU, reproduce exactly.

**A single experiment,** for example the main SAE on section B with seed 1:

```
SAEGT_SEED=1 python scripts/run_embeddings.py B tsae_cons,tsae,varimax_topk Expr50,CellCharter,CADAST
```

## Provenance: is this the code that ran?

The library modules are not rewritten. `antig/N2/tools/95_build_repo.py` in the authors' research workspace copies them function by function from the research code (`inspect.getsource`) and changes only import lines. The scripts were ported by `tools/97_port_scripts.py`, which applies explicit, asserted path and import substitutions; each script's header names its source. Branches not used in the paper were removed, as were inputs excluded after quality control (a zero-shot Novae run, an unpublished model).

`tests/test_equivalence.py` checks the result at four levels:
1. every library function, research version vs this repository, gives identical outputs on identical inputs;
2. the deposited inputs are identical to the arrays the research readers return;
3. rerunning the deterministic arms through these scripts reproduces the paper's run files exactly;
4. `make_tables.py` regenerates the paper's tables byte for byte.

On the authors' machine all 23 checks pass. Levels 3–4 run without the research code: `python tests/test_equivalence.py --quick`.

## Notes

- **Apple MPS top-k:** a single `topk` over more than about 2^28 elements returned 0 without an error on PyTorch MPS. `saegt/sae.py` therefore calibrates large BatchTopK thresholds from a 200,000-row CPU sample (Methods).
- **Consensus anchor:** the consensus is anchored on the first fit, so it is a subset of that fit (Methods, Discussion).

## License and citation

MIT (see `LICENSE`). Please cite the paper (see `CITATION.cff`).
