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

from evaluate_hybrid import load_model  # noqa: E402
from train_model import predict  # noqa: E402
from virtual_cell.data import load_train_val  # noqa: E402
from virtual_cell.features import FeatureEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check prediction-file contract without test targets")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=ROOT / "artifacts" / "experiments" / "balanced_monolithic_batch_multiloss",
    )
    return parser.parse_args()


def resolve_test_metadata(input_dir: Path) -> Path:
    matches = sorted(input_dir.glob("*metadata_test*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one test metadata file, found {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    data = load_train_val(args.input_dir)
    test_metadata = pd.read_csv(resolve_test_metadata(args.input_dir)).set_index("sample_ID")
    if not test_metadata.index.is_unique:
        raise AssertionError("Test sample_ID must be unique")

    encoder = FeatureEncoder.from_dict(
        json.loads((args.experiment_dir / "feature_encoder.json").read_text(encoding="utf-8"))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, experiment_config = load_model(
        args.experiment_dir, encoder, len(data.kept_proteins), device
    )
    prediction = predict(
        model,
        encoder.transform(test_metadata),
        device,
        int(experiment_config["batch_size"]) * 2,
    )

    assert prediction.shape[0] == len(test_metadata) == 4454
    assert prediction.shape[1] == len(data.kept_proteins) == 4422
    assert np.isfinite(prediction).all()
    assert test_metadata.index.notna().all()
    report = {
        "sample_rows": int(prediction.shape[0]),
        "protein_columns": int(prediction.shape[1]),
        "sample_id_unique": bool(test_metadata.index.is_unique),
        "has_na": bool(np.isnan(prediction).any()),
        "has_inf": bool(np.isinf(prediction).any()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "prediction_scale": "log2",
        "test_target_loaded": False,
    }
    output = args.experiment_dir / "submission_contract_check.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

