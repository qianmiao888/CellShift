from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_cell.data import load_train_val  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit official train/validation data")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "audit" / "data_audit.json")
    parser.add_argument("--missing-rate-threshold", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_train_val(args.input_dir, args.missing_rate_threshold)
    metadata = data.metadata
    raw = data.raw_proteome
    log2_data = data.log2_proteome

    split_counts = metadata["split_final"].value_counts().sort_index().to_dict()
    observed = log2_data.to_numpy()
    finite = observed[np.isfinite(observed)]
    report = {
        "input_dir": str(args.input_dir.resolve()),
        "metadata_rows": int(len(metadata)),
        "metadata_columns": metadata.columns.tolist(),
        "sample_id_unique": bool(metadata.index.is_unique and raw.index.is_unique),
        "sample_id_aligned": bool(metadata.index.equals(raw.index)),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "raw_protein_columns": int(raw.shape[1]),
        "missing_rate_threshold": float(args.missing_rate_threshold),
        "kept_proteins": int(len(data.kept_proteins)),
        "removed_proteins": int(raw.shape[1] - len(data.kept_proteins)),
        "observed_fraction_after_filter": float(data.observed_mask.to_numpy().mean()),
        "log2_min_observed": float(finite.min()),
        "log2_max_observed": float(finite.max()),
        "test_target_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
