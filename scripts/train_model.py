from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_cell.baselines import is_treatment, matched_control_prediction  # noqa: E402
from virtual_cell.data import load_train_val  # noqa: E402
from virtual_cell.features import FeatureEncoder  # noqa: E402
from virtual_cell.losses import (  # noqa: E402
    masked_column_pearson_loss,
    masked_mse,
    masked_row_pearson_loss,
)
from virtual_cell.metrics import diagnostic_metrics  # noqa: E402
from virtual_cell.model import LowRankConditionModel  # noqa: E402


class ProteomeDataset(Dataset):
    def __init__(
        self,
        features: dict[str, np.ndarray],
        target: np.ndarray,
        control: np.ndarray,
    ) -> None:
        self.features = {
            key: torch.from_numpy(value) for key, value in features.items()
        }
        self.mask = torch.from_numpy(np.isfinite(target))
        self.target = torch.from_numpy(np.nan_to_num(target, nan=0.0).astype(np.float32))
        self.control_mask = torch.from_numpy(np.isfinite(control))
        self.control = torch.from_numpy(np.nan_to_num(control, nan=0.0).astype(np.float32))

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.features.items()}
        item["target"] = self.target[index]
        item["mask"] = self.mask[index]
        item["control"] = self.control[index]
        item["control_mask"] = self.control_mask[index]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a closed-data low-rank model")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--architecture", choices=["monolithic", "residual"], default="residual")
    parser.add_argument("--use-batch-calibration", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--decoder-init", choices=["pca", "random"], default="pca")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--category-dropout", type=float, default=0.15)
    parser.add_argument("--fc-weight", type=float, default=0.0)
    parser.add_argument("--protein-corr-weight", type=float, default=0.0)
    parser.add_argument("--component-reg-weight", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pca_initialization(
    y_train: pd.DataFrame,
    latent_dim: int,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"pca_{y_train.shape[1]}p_{latent_dim}d.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["mean"], cached["components"], float(cached["explained"])

    values = y_train.to_numpy(dtype=np.float32, copy=True)
    protein_mean = np.nanmean(values, axis=0).astype(np.float32)
    filled = np.where(np.isfinite(values), values, protein_mean[None, :])
    centered = filled - protein_mean[None, :]
    pca = PCA(n_components=latent_dim, svd_solver="randomized", random_state=20260809)
    pca.fit(centered)
    components = pca.components_.astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum())
    np.savez_compressed(
        cache_path,
        mean=protein_mean,
        components=components,
        explained=np.asarray(explained, dtype=np.float32),
    )
    return protein_mean, components, explained


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def predict(
    model: LowRankConditionModel,
    features: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    total = len(next(iter(features.values())))
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = {
            key: torch.from_numpy(value[start:end]).to(device)
            for key, value in features.items()
        }
        prediction, _ = model(batch)
        output.append(prediction.cpu().numpy())
    return np.concatenate(output, axis=0)


def masked_global_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    xv = x[valid].astype(np.float64)
    yv = y[valid].astype(np.float64)
    xv -= xv.mean()
    yv -= yv.mean()
    denominator = np.sqrt(np.square(xv).sum() * np.square(yv).sum())
    return float((xv * yv).sum() / denominator) if denominator > 0 else float("nan")


def evaluate_scenes(
    metadata: pd.DataFrame,
    y: pd.DataFrame,
    validation_ids: pd.Index,
    prediction: np.ndarray,
    control_prediction: pd.DataFrame,
    validation_splits: list[str],
    control_labels: list[str],
    quality_control_label: str,
    full_metrics: bool,
) -> list[dict[str, float | int | str]]:
    pred_frame = pd.DataFrame(prediction, index=validation_ids, columns=y.columns)
    treatment_rows = is_treatment(metadata, control_labels, quality_control_label)
    results: list[dict[str, float | int | str]] = []
    for split in validation_splits:
        ids = metadata.index[
            metadata["split_final"].eq(split) & treatment_rows
        ]
        truth = y.loc[ids].to_numpy(dtype=np.float32, copy=False)
        pred = pred_frame.loc[ids].to_numpy(dtype=np.float32, copy=False)
        if full_metrics:
            metrics = diagnostic_metrics(truth, pred)
        else:
            valid = np.isfinite(truth) & np.isfinite(pred)
            metrics = {
                "log2_rmse": float(np.sqrt(np.square(truth[valid] - pred[valid]).mean())),
                "global_r2": float("nan"),
                "protein_r2_median": float("nan"),
            }
        control = control_prediction.loc[ids].to_numpy(dtype=np.float32, copy=False)
        delta_true = truth - control
        delta_pred = pred - control
        delta_valid = np.isfinite(delta_true) & np.isfinite(delta_pred)
        delta_rmse = float(
            np.sqrt(np.square(delta_true[delta_valid] - delta_pred[delta_valid]).mean())
        )
        results.append(
            {
                "split": split,
                "samples": int(len(ids)),
                **metrics,
                "delta_rmse": delta_rmse,
                "delta_pcc": masked_global_corr(delta_true, delta_pred),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["random_seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = ROOT / "artifacts" / "experiments" / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args) | {"device": str(device), "seed": seed}, default=str, indent=2),
        encoding="utf-8",
    )

    data = load_train_val(args.input_dir, config["missing_rate_threshold"])
    metadata = data.metadata
    y = data.log2_proteome
    train_ids = metadata.index[metadata["split_final"].eq("train")]
    validation_ids = metadata.index[metadata["split_final"].ne("train")]

    encoder = FeatureEncoder.fit(
        metadata.loc[train_ids],
        config["control_labels"],
        config["quality_control_label"],
    )
    (output_dir / "feature_encoder.json").write_text(
        json.dumps(encoder.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_features = encoder.transform(metadata.loc[train_ids])
    validation_features = encoder.transform(metadata.loc[validation_ids])

    train_control = matched_control_prediction(
        metadata.loc[train_ids],
        y.loc[train_ids],
        train_ids,
        config["control_labels"],
        config["matched_control_keys"],
    )
    evaluation_control = matched_control_prediction(
        metadata,
        y,
        validation_ids,
        config["control_labels"],
        config["matched_control_keys"],
    )

    protein_mean, pca_components, explained = pca_initialization(
        y.loc[train_ids], args.latent_dim, ROOT / "artifacts" / "cache"
    )
    model = LowRankConditionModel(
        cardinalities=encoder.cardinalities(),
        protein_mean=torch.from_numpy(protein_mean),
        decoder_initialization=torch.from_numpy(pca_components),
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        category_dropout=args.category_dropout,
        use_batch_calibration=args.use_batch_calibration,
        initialize_decoder=args.decoder_init == "pca",
    ).to(device)

    dataset = ProteomeDataset(
        train_features,
        y.loc[train_ids].to_numpy(dtype=np.float32, copy=False),
        train_control.to_numpy(dtype=np.float32, copy=False),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_epoch = 0
    patience_left = args.patience
    best_path = output_dir / "best_model.pt"

    print(
        f"device={device} proteins={y.shape[1]} latent={args.latent_dim} "
        f"pca_explained={explained:.4f}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total": 0.0, "mse": 0.0, "fc": 0.0, "protein": 0.0, "reg": 0.0}
        batches = 0
        for raw_batch in loader:
            batch = to_device(raw_batch, device)
            prediction, components = model(batch)
            loss_mse = masked_mse(prediction, batch["target"], batch["mask"])

            fc_mask = batch["mask"] & batch["control_mask"] & batch["is_treatment"].bool()
            delta_prediction = prediction - batch["control"]
            delta_target = batch["target"] - batch["control"]
            loss_fc = (
                masked_row_pearson_loss(delta_prediction, delta_target, fc_mask)
                if args.fc_weight > 0
                else prediction.new_zeros(())
            )
            loss_protein = (
                masked_column_pearson_loss(prediction, batch["target"], batch["mask"])
                if args.protein_corr_weight > 0
                else prediction.new_zeros(())
            )
            regularized = [
                value.square().mean()
                for key, value in components.items()
                if key in {"chemical", "interaction", "batch"}
            ]
            loss_reg = torch.stack(regularized).mean() if regularized else prediction.new_zeros(())
            loss = (
                loss_mse
                + args.fc_weight * loss_fc
                + args.protein_corr_weight * loss_protein
                + args.component_reg_weight * loss_reg
            )
            if not torch.isfinite(loss):
                diagnostics = {
                    "loss": float(loss.detach()),
                    "mse": float(loss_mse.detach()),
                    "fc": float(loss_fc.detach()),
                    "protein": float(loss_protein.detach()),
                    "reg": float(loss_reg.detach()),
                    "prediction_finite": bool(torch.isfinite(prediction).all()),
                    "target_finite": bool(torch.isfinite(batch["target"]).all()),
                }
                raise FloatingPointError(f"Non-finite training loss: {diagnostics}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            sums["total"] += float(loss.detach())
            sums["mse"] += float(loss_mse.detach())
            sums["fc"] += float(loss_fc.detach())
            sums["protein"] += float(loss_protein.detach())
            sums["reg"] += float(loss_reg.detach())
            batches += 1

        validation_prediction = predict(
            model, validation_features, device, args.batch_size * 2
        )
        scene_metrics = evaluate_scenes(
            metadata,
            y,
            validation_ids,
            validation_prediction,
            evaluation_control,
            config["validation_splits"],
            config["control_labels"],
            config["quality_control_label"],
            full_metrics=False,
        )
        balanced_rmse = float(np.mean([row["log2_rmse"] for row in scene_metrics]))
        balanced_delta_pcc = float(np.mean([row["delta_pcc"] for row in scene_metrics]))
        row = {
            "epoch": epoch,
            "train_loss": sums["total"] / batches,
            "train_mse": sums["mse"] / batches,
            "train_fc_loss": sums["fc"] / batches,
            "train_protein_corr_loss": sums["protein"] / batches,
            "balanced_val_rmse": balanced_rmse,
            "balanced_val_delta_pcc": balanced_delta_pcc,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={row['train_loss']:.4f} "
            f"val_rmse={balanced_rmse:.4f} val_delta_pcc={balanced_delta_pcc:.4f}"
        )

        if balanced_rmse < best_score - 1e-4:
            best_score = balanced_rmse
            best_epoch = epoch
            patience_left = args.patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_prediction = predict(model, validation_features, device, args.batch_size * 2)
    final_metrics = evaluate_scenes(
        metadata,
        y,
        validation_ids,
        final_prediction,
        evaluation_control,
        config["validation_splits"],
        config["control_labels"],
        config["quality_control_label"],
        full_metrics=True,
    )
    summary = {
        "experiment": args.experiment_name,
        "best_epoch": best_epoch,
        "best_balanced_rmse": best_score,
        "pca_explained_variance": explained,
        "metrics": final_metrics,
        "test_target_loaded": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
