from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_cell.baselines import (  # noqa: E402
    is_treatment,
    matched_control_prediction,
    protein_mean_prediction,
)
from virtual_cell.data import load_train_val  # noqa: E402
from virtual_cell.metrics import diagnostic_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe statistical baselines")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "baselines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = load_train_val(args.input_dir, config["missing_rate_threshold"])
    metadata = data.metadata
    y = data.log2_proteome

    rows: list[dict[str, float | int | str]] = []
    treatment_rows = is_treatment(
        metadata,
        config["control_labels"],
        config["quality_control_label"],
    )

    for split_name in config["validation_splits"]:
        scene_rows = metadata["split_final"].eq(split_name)
        split_rows = scene_rows & treatment_rows
        sample_ids = metadata.index[split_rows]
        if len(sample_ids) == 0:
            raise ValueError(f"No treatment rows found for split {split_name}")

        truth = y.loc[sample_ids]
        mean_prediction = protein_mean_prediction(metadata, y, sample_ids)
        control_prediction = matched_control_prediction(
            metadata,
            y,
            sample_ids,
            config["control_labels"],
            config["matched_control_keys"],
        )
        control_eligible = control_prediction.notna().any(axis=1)
        eligible_ids = sample_ids[control_eligible.to_numpy()]
        if len(eligible_ids) == 0:
            raise ValueError(f"No exact matched controls found for split {split_name}")

        truth_np = truth.loc[eligible_ids].to_numpy(dtype=np.float32, copy=False)
        predictions = {
            "protein_mean": mean_prediction.loc[eligible_ids],
            "matched_control": control_prediction.loc[eligible_ids],
        }
        for baseline_name, prediction in predictions.items():
            metrics = diagnostic_metrics(
                truth_np,
                prediction.to_numpy(dtype=np.float32, copy=False),
            )
            rows.append(
                {
                    "split": split_name,
                    "baseline": baseline_name,
                    "scene_samples_total": int(scene_rows.sum()),
                    "treatment_samples": int(split_rows.sum()),
                    "eligible_samples": int(len(eligible_ids)),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "baseline_diagnostics.csv"
    json_path = args.output_dir / "baseline_diagnostics.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(results.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
