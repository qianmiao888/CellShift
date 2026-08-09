from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_SPLIT = "train"
SAMPLE_ID = "sample_ID"


@dataclass(frozen=True)
class CompetitionData:
    metadata: pd.DataFrame
    raw_proteome: pd.DataFrame
    log2_proteome: pd.DataFrame
    observed_mask: pd.DataFrame
    kept_proteins: pd.Index
    train_missing_rate: pd.Series


def _resolve_single(input_dir: Path, pattern: str) -> Path:
    matches = sorted(input_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {input_dir}, "
            f"found {[path.name for path in matches]}"
        )
    return matches[0]


def _validate_sample_ids(frame: pd.DataFrame, source_name: str) -> None:
    if frame.index.hasnans:
        raise ValueError(f"{source_name} contains missing sample_ID values")
    if not frame.index.is_unique:
        duplicates = frame.index[frame.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"{source_name} contains duplicate sample_ID values: {duplicates}")


def load_train_val(
    input_dir: str | Path,
    missing_rate_threshold: float = 0.80,
) -> CompetitionData:
    """Load only the official train/validation files.

    The test target file is deliberately not resolved or read here. This makes the
    no-test-leakage boundary structural rather than dependent on operator discipline.
    """

    input_path = Path(input_dir).expanduser().resolve()
    metadata_path = _resolve_single(input_path, "*metadata_train_val*.csv")
    proteome_path = _resolve_single(input_path, "*proteome_raw_train_val*.csv")

    metadata = pd.read_csv(metadata_path).set_index(SAMPLE_ID, drop=True)
    proteome_columns = pd.read_csv(proteome_path, nrows=0).columns
    protein_dtypes = {
        column: np.float32 for column in proteome_columns if column != SAMPLE_ID
    }
    raw = pd.read_csv(
        proteome_path,
        index_col=SAMPLE_ID,
        dtype=protein_dtypes,
    )
    _validate_sample_ids(metadata, metadata_path.name)
    _validate_sample_ids(raw, proteome_path.name)

    missing_in_proteome = metadata.index.difference(raw.index)
    extra_in_proteome = raw.index.difference(metadata.index)
    if len(missing_in_proteome) or len(extra_in_proteome):
        raise ValueError(
            "Metadata/proteome sample_ID mismatch: "
            f"missing={len(missing_in_proteome)}, extra={len(extra_in_proteome)}"
        )
    raw = raw.loc[metadata.index]

    if "split_final" not in metadata.columns:
        raise ValueError("Metadata is missing required split_final column")
    train_rows = metadata["split_final"].eq(TRAIN_SPLIT)
    if not train_rows.any():
        raise ValueError("No rows with split_final == 'train'")

    train_missing_rate = raw.loc[train_rows].isna().mean(axis=0)
    kept_proteins = train_missing_rate.index[
        train_missing_rate.lt(missing_rate_threshold)
    ]
    filtered = raw.loc[:, kept_proteins]

    observed_values = filtered.to_numpy(copy=False)
    finite_observed = observed_values[np.isfinite(observed_values)]
    if finite_observed.size == 0 or np.any(finite_observed <= 0):
        raise ValueError("Observed raw intensities must be finite and strictly positive")

    with np.errstate(divide="ignore", invalid="ignore"):
        log2_values = np.log2(observed_values)
    log2_proteome = pd.DataFrame(
        log2_values,
        index=filtered.index,
        columns=filtered.columns,
    )
    observed_mask = log2_proteome.notna()

    return CompetitionData(
        metadata=metadata,
        raw_proteome=raw,
        log2_proteome=log2_proteome,
        observed_mask=observed_mask,
        kept_proteins=kept_proteins,
        train_missing_rate=train_missing_rate,
    )
