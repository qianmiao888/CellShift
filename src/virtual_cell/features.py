from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CATEGORICAL_FIELDS = (
    "Strains",
    "perturbation_no_concentration",
    "Medium",
    "Temperature",
    "data_source",
    "instrument",
    "Yeast_cell_plate",
)


@dataclass
class FeatureEncoder:
    vocabs: dict[str, dict[str, int]]
    max_time: float
    median_time: float
    control_labels: tuple[str, ...]
    quality_control_label: str

    @classmethod
    def fit(
        cls,
        train_metadata: pd.DataFrame,
        control_labels: list[str],
        quality_control_label: str,
    ) -> "FeatureEncoder":
        vocabs: dict[str, dict[str, int]] = {}
        for field in CATEGORICAL_FIELDS:
            values = sorted(train_metadata[field].astype(str).unique().tolist())
            vocabs[field] = {value: index + 1 for index, value in enumerate(values)}
        times = train_metadata["pert_time"].astype(float).to_numpy()
        return cls(
            vocabs=vocabs,
            max_time=float(times.max()),
            median_time=float(np.median(times)),
            control_labels=tuple(control_labels),
            quality_control_label=quality_control_label,
        )

    def transform(self, metadata: pd.DataFrame) -> dict[str, np.ndarray]:
        encoded: dict[str, np.ndarray] = {}
        for field, vocab in self.vocabs.items():
            encoded[field] = (
                metadata[field].astype(str).map(vocab).fillna(0).astype(np.int64).to_numpy()
            )

        time = metadata["pert_time"].astype(float).to_numpy(dtype=np.float32)
        scaled = time / max(self.max_time, 1.0)
        log_scaled = np.log1p(time) / np.log1p(max(self.max_time, 1.0))
        decay = np.exp(-time / max(self.median_time, 1.0))
        encoded["time_features"] = np.stack(
            [scaled, np.square(scaled), log_scaled, decay], axis=1
        ).astype(np.float32)

        excluded = [*self.control_labels, self.quality_control_label]
        encoded["is_treatment"] = (
            ~metadata["perturbation_no_concentration"].isin(excluded)
        ).to_numpy(dtype=np.float32)[:, None]
        return encoded

    def cardinalities(self) -> dict[str, int]:
        return {field: len(vocab) + 1 for field, vocab in self.vocabs.items()}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeatureEncoder":
        return cls(
            vocabs={
                str(field): {str(value): int(index) for value, index in vocab.items()}
                for field, vocab in payload["vocabs"].items()
            },
            max_time=float(payload["max_time"]),
            median_time=float(payload["median_time"]),
            control_labels=tuple(str(value) for value in payload["control_labels"]),
            quality_control_label=str(payload["quality_control_label"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "vocabs": self.vocabs,
            "max_time": self.max_time,
            "median_time": self.median_time,
            "control_labels": list(self.control_labels),
            "quality_control_label": self.quality_control_label,
        }
