from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.hest1k import (
    DEFAULT_TRAIN_IDS,
    DEFAULT_VAL_IDS,
    GeneStandardizer,
    HestSpotDataset,
    build_split_masks,
    load_processed_hest_data,
)
from src.models.phase1_baseline import build_model
from src.training.metrics import pearson_loss, summarize_predictions
from src.utils.paths import MODEL_DIR, PROCESSED_DIR
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 1 HEST1k baseline.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", choices=("metadata", "image", "multimodal"), default="multimodal")
    parser.add_argument("--train-backbone", choices=("all", "layer4", "frozen"), default="layer4")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=None, help="Optional resize before normalization.")
    parser.add_argument("--train-ids", nargs="+", default=list(DEFAULT_TRAIN_IDS))
    parser.add_argument("--val-ids", nargs="+", default=list(DEFAULT_VAL_IDS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pearson-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def limit_indices(indices: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=max_samples, replace=False)
    return np.sort(chosen)


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, GeneStandardizer, dict[str, Any]]:
    data = load_processed_hest_data(args.processed_dir)
    train_mask, val_mask = build_split_masks(data.sample_ids, args.train_ids, args.val_ids)

    train_indices = limit_indices(np.flatnonzero(train_mask), args.max_train_samples, seed=args.seed)
    val_indices = limit_indices(np.flatnonzero(val_mask), args.max_val_samples, seed=args.seed + 1)

    standardizer = GeneStandardizer().fit(data.genes[train_indices])
    train_genes = standardizer.transform(data.genes[train_indices])
    val_genes = standardizer.transform(data.genes[val_indices])

    train_dataset = HestSpotDataset(
        patches=data.patches[train_indices],
        genes=train_genes,
        metadata=data.metadata[train_indices],
        labels=data.labels[train_indices],
        sample_ids=data.sample_ids[train_indices],
        image_size=args.image_size,
    )
    val_dataset = HestSpotDataset(
        patches=data.patches[val_indices],
        genes=val_genes,
        metadata=data.metadata[val_indices],
        labels=data.labels[val_indices],
        sample_ids=data.sample_ids[val_indices],
        image_size=args.image_size,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    split_info = {
        "train_ids": list(args.train_ids),
        "val_ids": list(args.val_ids),
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "metadata_dim": int(data.metadata.shape[1]),
        "gene_dim": int(data.genes.shape[1]),
    }
    return train_loader, val_loader, standardizer, split_info


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    standardizer: GeneStandardizer,
    mse_loss: nn.Module,
    pearson_weight: float,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[str] = []

    with torch.no_grad():
        for images, genes, metadata, _labels, batch_sample_ids in data_loader:
            images = images.to(device, non_blocking=True)
            genes = genes.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            pred = model(images, metadata)
            loss = mse_loss(pred, genes) + pearson_weight * pearson_loss(pred, genes)
            losses.append(loss.item())

            preds.append(pred.cpu().numpy())
            targets.append(genes.cpu().numpy())
            sample_ids.extend(batch_sample_ids)

    pred_array = np.concatenate(preds, axis=0)
    target_array = np.concatenate(targets, axis=0)

    pred_original = standardizer.inverse_transform(pred_array)
    target_original = standardizer.inverse_transform(target_array)
    metrics = summarize_predictions(pred_original, target_original, np.asarray(sample_ids))
    metrics["loss"] = float(np.mean(losses))
    metrics["predictions"] = pred_original
    metrics["targets"] = target_original
    metrics["sample_ids"] = np.asarray(sample_ids)
    return metrics


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mse_loss: nn.Module,
    pearson_weight: float,
) -> float:
    model.train()
    running_loss = 0.0

    for images, genes, metadata, _labels, _sample_ids in data_loader:
        images = images.to(device, non_blocking=True)
        genes = genes.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(images, metadata)
        loss = mse_loss(pred, genes) + pearson_weight * pearson_loss(pred, genes)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(data_loader), 1)


def to_serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"predictions", "targets", "sample_ids"}}


def serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, standardizer, split_info = create_dataloaders(args)

    model = build_model(
        model_name=args.model,
        metadata_dim=split_info["metadata_dim"],
        gene_dim=split_info["gene_dim"],
        pretrained=args.pretrained,
        trainable_backbone=args.train_backbone,
    ).to(device)

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    mse_loss = nn.MSELoss()

    run_dir = args.output_dir / (args.run_name or args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable_config = serialize_config(args)

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_val_spot = float("-inf")
    epochs_without_improvement = 0

    print(f"Training {args.model} baseline on {device}")
    print(json.dumps(split_info, indent=2))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            mse_loss=mse_loss,
            pearson_weight=args.pearson_weight,
        )

        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            standardizer=standardizer,
            mse_loss=mse_loss,
            pearson_weight=args.pearson_weight,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **to_serializable_metrics(val_metrics),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Spot Pearson: {val_metrics['spot_pearson']:.4f} | "
            f"Val Gene Pearson: {val_metrics['gene_pearson_mean']:.4f}"
        )

        if val_metrics["spot_pearson"] > best_val_spot + args.min_delta:
            best_val_spot = val_metrics["spot_pearson"]
            epochs_without_improvement = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": deepcopy(model.state_dict()),
                "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                "config": serializable_config,
                "split_info": split_info,
                "metrics": to_serializable_metrics(val_metrics),
                "standardizer_mean": standardizer.mean,
                "standardizer_std": standardizer.std,
            }
            np.savez_compressed(
                run_dir / "best_val_predictions.npz",
                predictions=val_metrics["predictions"],
                targets=val_metrics["targets"],
                sample_ids=val_metrics["sample_ids"],
            )
            print("Saved new best checkpoint.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    torch.save(best_state, run_dir / "best_model.pt")

    with (run_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "config": serializable_config,
                "split_info": split_info,
                "best_epoch": best_state["epoch"],
                "best_metrics": best_state["metrics"],
            },
            fp,
            indent=2,
        )

    print("Best validation metrics:")
    print(json.dumps(best_state["metrics"], indent=2))
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
