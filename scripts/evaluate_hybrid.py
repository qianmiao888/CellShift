from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_model import evaluate_scenes, predict  # noqa: E402
from virtual_cell.baselines import is_treatment, matched_control_prediction  # noqa: E402
from virtual_cell.data import load_train_val  # noqa: E402
from virtual_cell.features import FeatureEncoder  # noqa: E402
from virtual_cell.metrics import diagnostic_metrics  # noqa: E402
from virtual_cell.model import LowRankConditionModel  # noqa: E402


BASE_KEYS = [
    "data_source",
    "instrument",
    "Yeast_cell_plate",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate neural/statistical hybrid and stress tests")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "balanced_monolithic_batch_multiloss",
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    return parser.parse_args()


def load_model(
    experiment_dir: Path,
    encoder: FeatureEncoder,
    protein_count: int,
    device: torch.device,
) -> tuple[LowRankConditionModel, dict[str, object]]:
    experiment_config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    latent_dim = int(experiment_config["latent_dim"])
    cache = np.load(ROOT / "artifacts" / "cache" / f"pca_{protein_count}p_{latent_dim}d.npz")
    model = LowRankConditionModel(
        cardinalities=encoder.cardinalities(),
        protein_mean=torch.from_numpy(cache["mean"].astype(np.float32)),
        decoder_initialization=torch.from_numpy(cache["components"].astype(np.float32)),
        architecture=str(experiment_config["architecture"]),
        hidden_dim=int(experiment_config["hidden_dim"]),
        dropout=float(experiment_config["dropout"]),
        category_dropout=float(experiment_config["category_dropout"]),
        use_batch_calibration=bool(experiment_config["use_batch_calibration"]),
        initialize_decoder=str(experiment_config.get("decoder_init", "pca")) == "pca",
    ).to(device)
    model.load_state_dict(
        torch.load(experiment_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return model, experiment_config


def build_factorized_prior(
    metadata: pd.DataFrame,
    y: pd.DataFrame,
    train_ids: pd.Index,
    validation_ids: pd.Index,
) -> pd.DataFrame:
    train_values = y.loc[train_ids].copy()
    train_values.index = pd.MultiIndex.from_frame(metadata.loc[train_ids, BASE_KEYS])
    base_profiles = train_values.groupby(
        level=list(range(len(BASE_KEYS))), sort=False, dropna=False
    ).mean()

    train_keys = pd.MultiIndex.from_frame(metadata.loc[train_ids, BASE_KEYS])
    base_train = base_profiles.reindex(train_keys)
    base_train.index = train_ids
    residual = y.loc[train_ids] - base_train
    strain_effect = residual.groupby(metadata.loc[train_ids, "Strains"]).mean()
    chemical_effect = residual.groupby(
        metadata.loc[train_ids, "perturbation_no_concentration"]
    ).mean()

    main_effect = np.nan_to_num(
        strain_effect.reindex(metadata.loc[train_ids, "Strains"]).to_numpy()
    ) + np.nan_to_num(
        chemical_effect.reindex(
            metadata.loc[train_ids, "perturbation_no_concentration"]
        ).to_numpy()
    )
    interaction_residual = pd.DataFrame(
        residual.to_numpy() - main_effect,
        index=pd.MultiIndex.from_frame(
            metadata.loc[train_ids, ["Strains", "perturbation_no_concentration"]]
        ),
        columns=y.columns,
    )
    interaction_effect = interaction_residual.groupby(level=[0, 1]).mean()

    validation_keys = pd.MultiIndex.from_frame(metadata.loc[validation_ids, BASE_KEYS])
    base_validation = base_profiles.reindex(validation_keys)
    base_validation.index = validation_ids
    strain_validation = strain_effect.reindex(metadata.loc[validation_ids, "Strains"]).to_numpy()
    chemical_validation = chemical_effect.reindex(
        metadata.loc[validation_ids, "perturbation_no_concentration"]
    ).to_numpy()
    pair_keys = pd.MultiIndex.from_frame(
        metadata.loc[validation_ids, ["Strains", "perturbation_no_concentration"]]
    )
    interaction_validation = interaction_effect.reindex(pair_keys).to_numpy()

    prediction = (
        base_validation.to_numpy()
        + np.nan_to_num(strain_validation)
        + np.nan_to_num(chemical_validation)
        + 0.5 * np.nan_to_num(interaction_validation)
    )
    return pd.DataFrame(prediction, index=validation_ids, columns=y.columns)


def evaluate_prediction(
    name: str,
    metadata: pd.DataFrame,
    y: pd.DataFrame,
    validation_ids: pd.Index,
    prediction: pd.DataFrame,
    control: pd.DataFrame,
    config: dict[str, object],
) -> list[dict[str, object]]:
    rows = evaluate_scenes(
        metadata,
        y,
        validation_ids,
        prediction.loc[validation_ids].to_numpy(dtype=np.float32, copy=False),
        control,
        config["validation_splits"],
        config["control_labels"],
        config["quality_control_label"],
        full_metrics=True,
    )
    for row in rows:
        row["method"] = name
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = load_train_val(args.input_dir, float(config["missing_rate_threshold"]))
    metadata, y = data.metadata, data.log2_proteome
    train_ids = metadata.index[metadata["split_final"].eq("train")]
    validation_ids = metadata.index[metadata["split_final"].ne("train")]

    encoder = FeatureEncoder.from_dict(
        json.loads((args.experiment_dir / "feature_encoder.json").read_text(encoding="utf-8"))
    )
    features = encoder.transform(metadata.loc[validation_ids])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, experiment_config = load_model(
        args.experiment_dir, encoder, y.shape[1], device
    )
    neural_array = predict(
        model, features, device, int(experiment_config["batch_size"]) * 2
    )
    neural = pd.DataFrame(neural_array, index=validation_ids, columns=y.columns)
    factorized = build_factorized_prior(metadata, y, train_ids, validation_ids)

    # Logic-driven router: a known strain with an unknown chemical, or a pure time
    # holdout, can use train-only factorized effects. Unknown-strain scenes use the
    # unknown-aware neural fallback.
    hybrid = neural.copy()
    for split in ("val_chem_only", "val_time"):
        ids = metadata.index[metadata["split_final"].eq(split)]
        hybrid.loc[ids] = factorized.loc[ids]

    control = matched_control_prediction(
        metadata,
        y,
        validation_ids,
        config["control_labels"],
        config["matched_control_keys"],
    )
    results: list[dict[str, object]] = []
    results.extend(
        evaluate_prediction("neural_multiloss", metadata, y, validation_ids, neural, control, config)
    )
    results.extend(
        evaluate_prediction("factorized_prior", metadata, y, validation_ids, factorized, control, config)
    )
    results.extend(
        evaluate_prediction("scenario_router", metadata, y, validation_ids, hybrid, control, config)
    )

    stress_results: list[dict[str, object]] = []
    for field in ("Yeast_cell_plate", "Strains", "perturbation_no_concentration"):
        stressed = {key: value.copy() for key, value in features.items()}
        stressed[field].fill(0)
        stressed_array = predict(
            model, stressed, device, int(experiment_config["batch_size"]) * 2
        )
        stressed_frame = pd.DataFrame(stressed_array, index=validation_ids, columns=y.columns)
        stress_results.extend(
            evaluate_prediction(
                f"zero_{field}", metadata, y, validation_ids, stressed_frame, control, config
            )
        )

    result_frame = pd.DataFrame(results)
    stress_frame = pd.DataFrame(stress_results)
    result_frame.to_csv(args.experiment_dir / "hybrid_metrics.csv", index=False)
    stress_frame.to_csv(args.experiment_dir / "stress_test_metrics.csv", index=False)
    summary = {
        "experiment": args.experiment_dir.name,
        "methods": results,
        "stress_tests": stress_results,
        "test_target_loaded": False,
    }
    (args.experiment_dir / "hybrid_and_stress_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = ["method", "split", "log2_rmse", "protein_r2_median", "delta_pcc"]
    print(result_frame[columns].to_string(index=False))
    print("\nStress tests:")
    print(stress_frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
