# Candidate Overlap and Query Utility in Low-Label Visual Entity Alignment

Core implementation for controlled entity-alignment and query-utility experiments. This repository contains source code only. Data, model weights, embeddings, predictions, and figures are generated locally.

## Environment

Python 3.12, Git, and a CUDA-capable GPU are required for the complete fixed CPU/GPU training schedule. Install a CUDA 12.4 build of PyTorch 2.6.0 using the official PyTorch instructions, then install `requirements.txt`. Analysis uses two CPU computation threads. Experiments were executed on Windows with an RTX 3090 Ti (24 GiB) and 32 GiB RAM. Full reproduction includes training. The frozen representations alone occupy approximately 12.55 GiB; allow additional space for raw data, checkpoints, and generated outcomes.

## Run

Run commands from the repository directory. Configuration constants and public-input checksums in `settings.py` are materialized automatically by the entry point.

```text
python -m pip install -r requirements.txt
python reproduce.py --stage plan
python reproduce.py --stage prepare
python reproduce.py --stage verify
python reproduce.py --stage train
python reproduce.py --stage evaluate
python reproduce.py --stage systems
python reproduce.py --stage revision
python reproduce.py --stage analysis
```

`prepare` retrieves public inputs, checks SHA-256 values, checks out pinned upstream versions, builds local compatibility copies, and reconstructs the deterministic split/mask manifest. It does not use supplied baseline predictions. `train` follows the fixed model/mask seeds and skips existing checkpoints. `evaluate` creates the main and candidate-continuation outcomes. `revision` runs actual retrieval, product-domain controls, duplicate-text sensitivity, two-hop queries, and fixed reciprocal-rank fusion. `systems` runs serial CPU timings. `analysis` regenerates tables, plots, and validation reports. Existing outcomes are protected; use `--fresh-results` with `evaluate` or `revision` to archive them before recomputing. Do not run concurrent timing jobs.

Cached predictions are not included: analysis requires prior stages or an existing locally generated result tree. End-to-end training was performed in the research environment. Release validation checks reconstructed manifests and computational replay; it does not constitute a second complete retraining on another machine.

## Inputs and upstream implementations

The multimodal graphs are FBDB15K, FBYG15K, and DBP15K (Chinese-English and French-English). Public data locations, checksums, split seeds, budgets, and query settings are recorded in `settings.py`. Original MMKB features are reconciled with the processed FBDB features before training. Exact duplicated visual features and non-bijective reference endpoints follow the declared eligibility rule. Original Abt-Buy and Amazon-Google tables come from Leipzig; product experiments use text only.

- [DESAlign](https://github.com/wyy-code/DESAlign), pinned to `d22eb74130c77d5c6bb29b1e47e5ee7286400155`.
- [CDMEA](https://github.com/sutaoyu/CDMEA), pinned to `c9929cc143d32ea6b9af97069c6f1d43c206c08d`.
- [Leipzig entity-resolution datasets](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution).

Upstream code and datasets retain their original terms. Preparation records compatibility changes, including equivalent sparse-gradient and attention evaluation and native PyTorch scatter operations. TFP is a common-embedding structural control with a numerical fidelity check, not an independently trained reproduction of its published headline results.

## Experimental interface

Row decisions, edge-constrained partial injection, and assignment over pooled targets use the same reranker. Missing true candidate edges are never inserted. Candidate-induced abstention is represented by -1 and omitted from answer sets. Queries use distinct set semantics; TP, FP, FN, micro F1, and identity accuracy are recorded separately. Validators independently recompute set answers and SQLite joins.

The main study uses 128 training correspondences and fixed development/test boundaries. The 256-source retrieval cohort and later 64-source cohort are distinguished in output filenames. Later mechanism and rank-fusion analyses are post-hoc diagnostics on already examined cohorts. Rank fusion fixes reciprocal-rank constant 60 and evaluates every registered k. No test-based best-k or best-checkpoint selection is implemented.

Timing separates scoring, decoding, materialization, and query execution. Retrieved-candidate experiments still construct the common dense reranking matrix; they do not demonstrate sublinear end-to-end search. The four graph pairs share underlying populations. The supplemental product datasets do not establish independent visual generalization.
