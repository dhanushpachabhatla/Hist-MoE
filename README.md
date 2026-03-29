# histMOE

Preliminary GSoC work on predicting spatial gene expression vectors from histology image patches.

The current source of truth is the scripted pipeline under `src/`. The notebooks are kept only as exploratory history and should not be treated as the main implementation.

## Project Goal

Build a reproducible baseline for histology-to-gene prediction on HEST1k, then test whether a small Mixture-of-Experts model improves over a strong single-head baseline.

## Current Status

The repo currently supports:

- HEST1k processed-array loading
- reproducible train/validation splits
- train-only target standardization
- ImageNet-normalized image inputs
- single-head baseline training
- minimal MoE training
- full-validation metric aggregation
- per-slide reporting
- saved checkpoints, prediction dumps, and JSON summaries

Current HEST1k setup:

- dataset: 6 HEST1k WSIs
- processed samples: 19,413 spots
- target genes: 150
- train slides: `TENX155`, `TENX175`, `TENX171`, `TENX189`
- validation slides: `TENX156`, `TENX190`

## Best Results So Far

All results below are from the scripted pipeline on the same split.

Single-head baselines:

- metadata-only: spot Pearson `0.0724`, gene Pearson `-0.0147`
- image-only: spot Pearson `0.0818`, gene Pearson `0.0237`
- image + metadata: spot Pearson `0.0835`, gene Pearson `0.0288`

MoE variants:

- minimal MoE: spot Pearson `0.0795`, gene Pearson `0.0263`
- minimal MoE + full backbone fine-tuning: spot Pearson `0.0808`, gene Pearson `0.0270`
- minimal MoE + image-only gate: spot Pearson `0.0966`, gene Pearson `0.0440`

Current best model:

- 2-expert MoE
- shared ResNet18 image encoder
- shared metadata encoder
- gate uses image features only
- experts use image + metadata
- full backbone fine-tuning

Current takeaway:

- metadata alone is not enough
- image carries most of the useful signal
- letting the gate see metadata hurts routing
- image-only gating is the first MoE variant that clearly beats the single-head baseline on the shared validation split
- `TENX190` remains the hardest validation slide

## Repository Layout

```text
histMOE/
|-- src/
|   |-- data/
|   |   `-- hest1k.py
|   |-- models/
|   |   |-- phase1_baseline.py
|   |   `-- phase2_moe.py
|   |-- training/
|   |   |-- train_phase1_baseline.py
|   |   `-- train_phase2_moe.py
|   `-- utils/
|       |-- paths.py
|       `-- reproducibility.py
|-- models/
|   `-- phase1_baseline/
|-- data300/
|   `-- processed/
|-- hest1k-dataset/
|-- notebooks/
`-- download_hest1k.py
```

## Main Scripts

Phase 1 single-head baselines:

- `python -m src.training.train_phase1_baseline --model metadata --run-name phase1_metadata_v1`
- `python -m src.training.train_phase1_baseline --model image --run-name phase1_image_v1`
- `python -m src.training.train_phase1_baseline --model multimodal --run-name phase1_multimodal_v1`

Phase 2 MoE:

- `python -m src.training.train_phase2_moe --run-name minimal_moe_v1`
- `python -m src.training.train_phase2_moe --run-name minimal_moe_v2_all_backbone --train-backbone all`
- `python -m src.training.train_phase2_moe --run-name minimal_moe_v3_image_gate --gate-input image --train-backbone all`

Windows examples with the local environment:

```powershell
.\venv\Scripts\python.exe -m src.training.train_phase1_baseline --model multimodal --run-name phase1_multimodal_v1
.\venv\Scripts\python.exe -m src.training.train_phase2_moe --run-name minimal_moe_v3_image_gate --gate-input image --train-backbone all
```

## Inputs And Outputs

Expected processed inputs in `data300/processed`:

- `patches.npy`
- `genes.npy`
- `metadata.npy`
- `labels.npy`
- `sample_ids.npy`

Training outputs are written under `models/phase1_baseline/...` and include:

- `best_model.pt`
- `summary.json`
- `history.json`
- `best_val_predictions.npz`

## Model Details

### Phase 1 baseline

- ResNet18 image encoder
- small metadata encoder
- single regression head
- loss: `MSE + 0.1 * PearsonLoss`

### Phase 2 minimal MoE

- shared ResNet18 image encoder
- shared metadata encoder
- 2 experts
- soft gating
- loss: `MSE + 0.1 * PearsonLoss + small load-balancing loss`

Best-performing routing design so far:

- gate uses image features only
- experts use image + metadata

## Notes On Notebooks

The notebooks are still in the repo because they document the exploratory path of the project, including early preprocessing and baseline attempts.

They are not the main implementation anymore.

If you want to reproduce or extend the current baseline, use the scripts in `src/`.

## Environment

Main dependencies used in the current pipeline:

- `numpy`
- `torch`
- `torchvision`
- `scanpy`
- `anndata`
- `scipy`
- `Pillow`
- `openslide-python`
- `scikit-learn`
- `tqdm`

## Next Steps

- run a cleaner comparison table across all baseline and MoE variants
- test whether `3 experts` helps now that image-only gating works
- improve generalization on `TENX190`
- move preprocessing from notebooks into a scripted `src/` pipeline as well
