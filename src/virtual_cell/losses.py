from __future__ import annotations

import torch


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(prediction.dtype)
    return ((prediction - target).square() * valid).sum() / valid.sum().clamp_min(1.0)


def masked_row_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    minimum_count: int = 16,
) -> torch.Tensor:
    valid = mask.to(prediction.dtype)
    counts = valid.sum(dim=1)
    usable = counts.ge(float(minimum_count))
    if not usable.any():
        return prediction.new_zeros(())

    denominator = counts.clamp_min(1.0)
    prediction_mean = (prediction * valid).sum(dim=1) / denominator
    target_mean = (target * valid).sum(dim=1) / denominator
    prediction_centered = (prediction - prediction_mean[:, None]) * valid
    target_centered = (target - target_mean[:, None]) * valid
    covariance = (prediction_centered * target_centered).sum(dim=1)
    prediction_norm = (prediction_centered.square().sum(dim=1) + 1e-8).sqrt()
    target_norm = (target_centered.square().sum(dim=1) + 1e-8).sqrt()
    correlation = covariance / (prediction_norm * target_norm).clamp_min(1e-8)
    return 1.0 - correlation[usable].mean()


def masked_column_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    minimum_count: int = 4,
) -> torch.Tensor:
    valid = mask.to(prediction.dtype)
    counts = valid.sum(dim=0)
    usable = counts.ge(float(minimum_count))
    if not usable.any():
        return prediction.new_zeros(())

    denominator = counts.clamp_min(1.0)
    prediction_mean = (prediction * valid).sum(dim=0) / denominator
    target_mean = (target * valid).sum(dim=0) / denominator
    prediction_centered = (prediction - prediction_mean[None, :]) * valid
    target_centered = (target - target_mean[None, :]) * valid
    covariance = (prediction_centered * target_centered).sum(dim=0)
    prediction_norm = (prediction_centered.square().sum(dim=0) + 1e-8).sqrt()
    target_norm = (target_centered.square().sum(dim=0) + 1e-8).sqrt()
    correlation = covariance / (prediction_norm * target_norm).clamp_min(1e-8)
    return 1.0 - correlation[usable].mean()
