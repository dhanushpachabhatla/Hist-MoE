from __future__ import annotations

from typing import Any

import numpy as np
import torch


def spotwise_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = pred_centered.norm(dim=1) * target_centered.norm(dim=1) + 1e-8
    return numerator / denominator


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - spotwise_pearson(pred, target).mean()


def gene_wise_pearson(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    correlations = np.zeros(pred.shape[1], dtype=np.float32)

    for gene_idx in range(pred.shape[1]):
        pred_gene = pred[:, gene_idx] - pred[:, gene_idx].mean()
        target_gene = target[:, gene_idx] - target[:, gene_idx].mean()
        denominator = np.linalg.norm(pred_gene) * np.linalg.norm(target_gene)

        if denominator > 1e-8:
            correlations[gene_idx] = float((pred_gene * target_gene).sum() / denominator)

    return correlations


def summarize_predictions(pred: np.ndarray, target: np.ndarray, sample_ids: np.ndarray) -> dict[str, Any]:
    pred_tensor = torch.from_numpy(pred)
    target_tensor = torch.from_numpy(target)

    overall_spot = float(spotwise_pearson(pred_tensor, target_tensor).mean().item())
    gene_corr = gene_wise_pearson(pred, target)

    metrics: dict[str, Any] = {
        "spot_pearson": overall_spot,
        "gene_pearson_mean": float(gene_corr.mean()),
        "gene_pearson_median": float(np.median(gene_corr)),
        "gene_pearson_positive_genes": int((gene_corr > 0).sum()),
        "num_samples": int(len(sample_ids)),
        "per_slide": {},
    }

    for sample_id in sorted(np.unique(sample_ids)):
        mask = sample_ids == sample_id
        slide_pred = pred[mask]
        slide_target = target[mask]
        slide_spot = float(spotwise_pearson(torch.from_numpy(slide_pred), torch.from_numpy(slide_target)).mean().item())
        slide_gene_corr = gene_wise_pearson(slide_pred, slide_target)

        metrics["per_slide"][str(sample_id)] = {
            "spot_pearson": slide_spot,
            "gene_pearson_mean": float(slide_gene_corr.mean()),
            "gene_pearson_positive_genes": int((slide_gene_corr > 0).sum()),
            "num_samples": int(mask.sum()),
        }

    return metrics
