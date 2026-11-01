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
python reproduce.py --stage retrieval
python reproduce.py --stage analysis
```

`prepare` retrieves public inputs, checks SHA-256 values, checks out pinned upstream versions, builds local compatibility copies, and reconstructs the deterministic split/mask manifest. It does not use supplied baseline predictions. `train` follows the fixed model/mask seeds and skips existing checkpoints. `evaluate` creates the main and candidate-continuation outcomes. `retrieval` runs actual retrieval, product-domain controls, duplicate-text sensitivity, two-hop queries, and fixed reciprocal-rank fusion. `systems` runs serial CPU timings. `analysis` regenerates tables, plots, and validation reports. Existing outcomes are protected; use `--fresh-results` with `evaluate` or `retrieval` to archive them before recomputing. Do not run concurrent timing jobs.

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


## Manuscript visual evidence

After the main evaluation and retrieval analyses, run `python build_visual_evidence.py`. This generates the checked motivating example, evaluation framework, assignment rewrite visualization, and a real-image illustration linked to recorded FR-EN predictions. It requires the frozen embeddings and retrieval outcomes generated by the previous stages. The public [DBP15K raw-case archive](https://huggingface.co/datasets/HackCz/DBP15K_raw/blob/main/DBP_raw.zip), linked by [MEAformer](https://github.com/zjukg/MEAformer), is downloaded separately and checked by SHA-256. Images and generated files are not stored in this repository.

The real-image case is selected by availability in that public archive and includes both declared model seeds. Archive-member hashes and all inspected decisions are generated in `visual/real_case.json`. Exact pixel correspondence between these public illustrations and the original cached encoder inputs is unverified; the images are not re-encoded and are not new experimental evidence of visual recognition quality. The motivating graph is explicitly a constructed example; the rewrite chart uses recorded outcomes.

To regenerate the detailed candidate-support diagram after acquiring the public case archive, run `python build_framework.py`. Its three-source graph is an explanatory schematic, not a measured result; the photographs illustrate public entities.

To redraw the entity/predicate motivation diagram and the image/decision case table, run `python build_study_figures.py` after the visual-evidence command. These figures use Times New Roman when available, with Matplotlib STIXGeneral as the fallback. Their PDFs embed the fonts. The case table displays all twelve recorded seed/decoder decisions; the displayed error photograph belongs only to the seed-one visual-pool prediction.

## Query execution scope and retail images

This experiment separates candidate provenance, source competition, and query execution scope. Its locally locked protocols are embedded in `settings.py`; they are not external preregistrations. Previously unscored graph sources remain part of transductive graphs. CDMEA-derived candidate generation was specified after the first scope results, before its own outcomes. The retail experiment is an independent image population with a derived catalog-linking protocol, not the dataset's original classification task.

After the main preparation, training, and evaluation commands above:

```text
python reproduce_scope.py --stage all
```

The stages `retail`, `graph`, `validate`, and `analysis` can be run separately. `retail` needs no graph training or existing predictions: it downloads the pinned public image release and CLIP checkpoint, checks hashes, encodes the images, fits the fixed residual ridge model, and evaluates all 128 retail conditions. `graph` requires the checkpoints and frozen embeddings from the main stages and produces 544 conditions. `validate` and `analysis` require both populations. All outcomes are generated in `query_scope`; existing condition files are protected and reused only with matching protocol hashes. To recompute, move that entire output directory to a dated archive first. Do not delete the original results to select a favorable run.

Retail inputs are from [Retail Visual RAG](https://huggingface.co/datasets/blamm/retail_visual_rag_pipeline), revision `bfb2f6e411b2e0976c899bb5618eb629f94c7b04`, associated with the CVPR 2025 FGVC workshop paper by Lamm and Keuper. The upstream dataset has a CC-BY-NC-ND-4.0 license; it is downloaded from its publisher and is not redistributed here. We use all 1,101 official-test images, with 64 fit, 64 development, and 239 evaluation groups. Two fitting views give 128 correspondences. Both test views share identities and are resampled together. Equivalence-group labels do not establish exact GTIN identity. Hash checks and decoded-pixel duplicate checks precede evaluation.

The encoder is OpenAI CLIP ViT-B/32, with QuickGELU and FP32, loaded from [this pinned checkpoint](https://huggingface.co/timm/vit_base_patch32_clip_224.openai/tree/a6f597a30f7b82c51704746581f9a4e41421e878). Set `ICDE_CLIP_CHECKPOINT` to an existing copy to avoid downloading it again; the same SHA-256 is required. Otherwise it downloads under `public_data/models`. Place this repository on a data drive. Feature extraction uses small batches and waits for an idle NVIDIA GPU; it does not stop other processes. The CPU analyses use two computation threads.

The graph primary setting is k=16; retail is k=4. All k=1,4,16,64 outcomes remain available. Row-local and classical component-closure executions are invariant controls. Permutation checks distinguish equal-objective alternative optima from solver errors. Disagreement is measured against a fixed full-source mapping, not automatically against ground truth. Pooling's retail quality gain fails to replicate; no new optimizer or visual encoder is claimed.

Output files include individual condition JSON/NPZ records, manifests, `aggregate.csv`, `aggregate.json`, `validation.json`, `permutation_audit.json`, and the vector figure `paper/figures/scope.pdf`. No model weights, images, results, or logs are included in this source-only package. Release validation includes an earlier separate-directory image re-encoding check and a packaging check that refits and re-evaluates all 128 retail conditions from the fixed image features. These checks are not an independent scientific replication or complete fresh-machine retraining of the graph models.

## Reproduction integrity

Configuration files are generated from `settings.py` with stable LF line endings. Preserve the generated protocols together with each result tree. Configuration hashes identify those exact files, including descriptive text; do not mix outcomes from a different protocol file. Numerical settings, supervision budgets, cohort boundaries, and random seeds are fixed. The named stages distinguish development diagnostics, the locked main evaluation, and later query-scope experiments. Local protocol locking is not external preregistration.
