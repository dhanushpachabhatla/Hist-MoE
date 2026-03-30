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

from src.data.hest1k import GeneStandardizer, HestSpotDataset, ProcessedHestData, load_processed_hest_data
from src.models.phase1_baseline import build_model
from src.models.phase2_moe import MinimalMoERegressor, load_balancing_loss
from src.training.metrics import pearson_loss, summarize_predictions
from src.utils.paths import MODEL_DIR, PROCESSED_DIR
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run slide-aware validation on HEST1k.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "slide_validation")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-kind", choices=("baseline", "moe"), required=True)
    parser.add_argument("--baseline-model", choices=("metadata", "image", "multimodal"), default="multimodal")
    parser.add_argument("--train-backbone", choices=("all", "layer4", "frozen"), default="all")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=None)
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
    parser.add_argument("--held-out-slides", nargs="+", default=None)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gate-input", choices=("multimodal", "image"), default="image")
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--load-balance-weight", type=float, default=0.01)
    return parser.parse_args()


def serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def limit_indices(indices: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=max_samples, replace=False)
    return np.sort(chosen)


def make_dataloaders(
    patches: np.ndarray,
    genes: np.ndarray,
    metadata: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    train_ids: list[str],
    val_ids: list[str],
    image_size: int | None,
    batch_size: int,
    num_workers: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    seed: int,
) -> tuple[DataLoader, DataLoader, GeneStandardizer, dict[str, Any]]:
    train_mask = np.isin(sample_ids, train_ids)
    val_mask = np.isin(sample_ids, val_ids)

    if not train_mask.any():
        raise ValueError("Slide-aware validation produced an empty training split.")
    if not val_mask.any():
        raise ValueError("Slide-aware validation produced an empty validation split.")

    train_indices = limit_indices(np.flatnonzero(train_mask), max_train_samples, seed)
    val_indices = limit_indices(np.flatnonzero(val_mask), max_val_samples, seed + 1)

    standardizer = GeneStandardizer().fit(genes[train_indices])
    train_genes = standardizer.transform(genes[train_indices])
    val_genes = standardizer.transform(genes[val_indices])

    train_dataset = HestSpotDataset(
        patches=patches[train_indices],
        genes=train_genes,
        metadata=metadata[train_indices],
        labels=labels[train_indices],
        sample_ids=sample_ids[train_indices],
        image_size=image_size,
    )
    val_dataset = HestSpotDataset(
        patches=patches[val_indices],
        genes=val_genes,
        metadata=metadata[val_indices],
        labels=labels[val_indices],
        sample_ids=sample_ids[val_indices],
        image_size=image_size,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    split_info = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "metadata_dim": int(metadata.shape[1]),
        "gene_dim": int(genes.shape[1]),
    }
    return train_loader, val_loader, standardizer, split_info


def evaluate_baseline(
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


def train_baseline_epoch(
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


def moe_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gate_weights: torch.Tensor,
    mse_loss: nn.Module,
    pearson_weight: float,
    load_balance_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_main = mse_loss(prediction, target)
    loss_corr = pearson_loss(prediction, target)
    loss_balance = load_balancing_loss(gate_weights)
    total = loss_main + pearson_weight * loss_corr + load_balance_weight * loss_balance
    return total, {
        "mse": float(loss_main.item()),
        "pearson_loss": float(loss_corr.item()),
        "load_balance_loss": float(loss_balance.item()),
    }


def evaluate_moe(
    model: MinimalMoERegressor,
    data_loader: DataLoader,
    device: torch.device,
    standardizer: GeneStandardizer,
    mse_loss: nn.Module,
    pearson_weight: float,
    load_balance_weight: float,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    mse_values: list[float] = []
    pearson_values: list[float] = []
    load_balance_values: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    sample_ids: list[str] = []

    with torch.no_grad():
        for images, genes, metadata, _labels, batch_sample_ids in data_loader:
            images = images.to(device, non_blocking=True)
            genes = genes.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            pred, gate_weights = model(images, metadata)
            loss, parts = moe_loss(pred, genes, gate_weights, mse_loss, pearson_weight, load_balance_weight)
            losses.append(loss.item())
            mse_values.append(parts["mse"])
            pearson_values.append(parts["pearson_loss"])
            load_balance_values.append(parts["load_balance_loss"])
            preds.append(pred.cpu().numpy())
            targets.append(genes.cpu().numpy())
            gates.append(gate_weights.cpu().numpy())
            sample_ids.extend(batch_sample_ids)

    pred_array = np.concatenate(preds, axis=0)
    target_array = np.concatenate(targets, axis=0)
    gate_array = np.concatenate(gates, axis=0)
    sample_array = np.asarray(sample_ids)

    pred_original = standardizer.inverse_transform(pred_array)
    target_original = standardizer.inverse_transform(target_array)
    metrics = summarize_predictions(pred_original, target_original, sample_array)
    metrics["loss"] = float(np.mean(losses))
    metrics["mse"] = float(np.mean(mse_values))
    metrics["pearson_loss"] = float(np.mean(pearson_values))
    metrics["load_balance_loss"] = float(np.mean(load_balance_values))
    metrics["gate_mean"] = gate_array.mean(axis=0).tolist()
    metrics["gate_per_slide"] = {
        str(sample_id): gate_array[sample_array == sample_id].mean(axis=0).tolist()
        for sample_id in sorted(np.unique(sample_array))
    }
    metrics["predictions"] = pred_original
    metrics["targets"] = target_original
    metrics["sample_ids"] = sample_array
    metrics["gate_weights"] = gate_array
    return metrics


def train_moe_epoch(
    model: MinimalMoERegressor,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mse_loss: nn.Module,
    pearson_weight: float,
    load_balance_weight: float,
) -> float:
    model.train()
    total_loss = 0.0

    for images, genes, metadata, _labels, _sample_ids in data_loader:
        images = images.to(device, non_blocking=True)
        genes = genes.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred, gate_weights = model(images, metadata)
        loss, _ = moe_loss(pred, genes, gate_weights, mse_loss, pearson_weight, load_balance_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(data_loader), 1)


def to_serializable_metrics(metrics: dict[str, Any], model_kind: str) -> dict[str, Any]:
    excluded = {"predictions", "targets", "sample_ids"}
    if model_kind == "moe":
        excluded.add("gate_weights")
    return {key: value for key, value in metrics.items() if key not in excluded}


def train_fold(
    args: argparse.Namespace,
    data: ProcessedHestData,
    all_slides: list[str],
    fold_index: int,
    held_out_slide: str,
) -> dict[str, Any]:
    train_ids = [slide for slide in all_slides if slide != held_out_slide]
    val_ids = [held_out_slide]

    fold_seed = args.seed + fold_index
    set_seed(fold_seed)

    train_loader, val_loader, standardizer, split_info = make_dataloaders(
        patches=data.patches,
        genes=data.genes,
        metadata=data.metadata,
        labels=data.labels,
        sample_ids=data.sample_ids,
        train_ids=train_ids,
        val_ids=val_ids,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        seed=fold_seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_loss = nn.MSELoss()

    if args.model_kind == "baseline":
        model = build_model(
            model_name=args.baseline_model,
            metadata_dim=split_info["metadata_dim"],
            gene_dim=split_info["gene_dim"],
            pretrained=args.pretrained,
            trainable_backbone=args.train_backbone,
        ).to(device)
    else:
        model = MinimalMoERegressor(
            metadata_dim=split_info["metadata_dim"],
            gene_dim=split_info["gene_dim"],
            num_experts=args.num_experts,
            gate_temperature=args.gate_temperature,
            gate_input=args.gate_input,
            pretrained=args.pretrained,
            trainable_backbone=args.train_backbone,
            dropout=args.dropout,
        ).to(device)

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state: dict[str, Any] | None = None
    best_val_spot = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    print(f"\n=== Fold {fold_index + 1}: hold out {held_out_slide} ===")
    print(json.dumps(split_info, indent=2))

    for epoch in range(1, args.epochs + 1):
        if args.model_kind == "baseline":
            train_loss = train_baseline_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                mse_loss=mse_loss,
                pearson_weight=args.pearson_weight,
            )
            val_metrics = evaluate_baseline(
                model=model,
                data_loader=val_loader,
                device=device,
                standardizer=standardizer,
                mse_loss=mse_loss,
                pearson_weight=args.pearson_weight,
            )
        else:
            train_loss = train_moe_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                mse_loss=mse_loss,
                pearson_weight=args.pearson_weight,
                load_balance_weight=args.load_balance_weight,
            )
            val_metrics = evaluate_moe(
                model=model,
                data_loader=val_loader,
                device=device,
                standardizer=standardizer,
                mse_loss=mse_loss,
                pearson_weight=args.pearson_weight,
                load_balance_weight=args.load_balance_weight,
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **to_serializable_metrics(val_metrics, args.model_kind),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
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
                "split_info": split_info,
                "metrics": to_serializable_metrics(val_metrics, args.model_kind),
                "standardizer_mean": standardizer.mean,
                "standardizer_std": standardizer.std,
                "history": history,
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError(f"Fold for held-out slide {held_out_slide} did not produce a checkpoint.")

    return {
        "held_out_slide": held_out_slide,
        "best_epoch": best_state["epoch"],
        "split_info": best_state["split_info"],
        "best_metrics": best_state["metrics"],
        "history": history,
    }


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    spot_scores = [fold["best_metrics"]["spot_pearson"] for fold in folds]
    gene_scores = [fold["best_metrics"]["gene_pearson_mean"] for fold in folds]

    return {
        "mean_spot_pearson": float(np.mean(spot_scores)),
        "std_spot_pearson": float(np.std(spot_scores)),
        "mean_gene_pearson": float(np.mean(gene_scores)),
        "std_gene_pearson": float(np.std(gene_scores)),
        "num_folds": len(folds),
        "per_fold": {fold["held_out_slide"]: fold["best_metrics"] for fold in folds},
    }


def main() -> None:
    args = parse_args()
    serializable_config = serialize_config(args)

    data = load_processed_hest_data(args.processed_dir)
    all_slides = sorted(np.unique(data.sample_ids).tolist())
    held_out_slides = args.held_out_slides or all_slides

    invalid = sorted(set(held_out_slides) - set(all_slides))
    if invalid:
        raise ValueError(f"Unknown held-out slides requested: {invalid}")

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    folds: list[dict[str, Any]] = []
    for fold_index, held_out_slide in enumerate(held_out_slides):
        fold_result = train_fold(args, data, all_slides, fold_index, held_out_slide)
        folds.append(fold_result)

    aggregate = summarize_folds(folds)
    payload = {
        "config": serializable_config,
        "held_out_slides": held_out_slides,
        "aggregate": aggregate,
        "folds": folds,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print("\n=== Slide-Aware Summary ===")
    print(json.dumps(aggregate, indent=2))
    print(f"Saved slide-aware validation summary to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
