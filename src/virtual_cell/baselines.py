from __future__ import annotations

import numpy as np
import pandas as pd


def is_control(metadata: pd.DataFrame, control_labels: list[str]) -> pd.Series:
    return metadata["perturbation_no_concentration"].isin(control_labels)


def is_treatment(
    metadata: pd.DataFrame,
    control_labels: list[str],
    quality_control_label: str,
) -> pd.Series:
    excluded = [*control_labels, quality_control_label]
    return ~metadata["perturbation_no_concentration"].isin(excluded)


def protein_mean_prediction(
    metadata: pd.DataFrame,
    log2_proteome: pd.DataFrame,
    sample_ids: pd.Index,
) -> pd.DataFrame:
    train_rows = metadata["split_final"].eq("train")
    train_mean = log2_proteome.loc[train_rows].mean(axis=0, skipna=True)
    prediction = np.repeat(train_mean.to_numpy()[None, :], len(sample_ids), axis=0)
    return pd.DataFrame(prediction, index=sample_ids, columns=log2_proteome.columns)


def matched_control_prediction(
    metadata: pd.DataFrame,
    log2_proteome: pd.DataFrame,
    sample_ids: pd.Index,
    control_labels: list[str],
    match_keys: list[str],
) -> pd.DataFrame:
    missing_keys = [key for key in match_keys if key not in metadata.columns]
    if missing_keys:
        raise ValueError(f"Matched-control keys absent from metadata: {missing_keys}")

    control_rows = is_control(metadata, control_labels)
    control_ids = metadata.index[control_rows]
    control_values = log2_proteome.loc[control_ids].copy()
    control_values.index = pd.MultiIndex.from_frame(
        metadata.loc[control_ids, match_keys], names=match_keys
    )
    control_profiles = control_values.groupby(
        level=list(range(len(match_keys))), sort=False, dropna=False
    ).mean()

    target_keys = pd.MultiIndex.from_frame(
        metadata.loc[sample_ids, match_keys], names=match_keys
    )
    prediction = control_profiles.reindex(target_keys)
    prediction.index = sample_ids
    prediction.columns = log2_proteome.columns
    return prediction

