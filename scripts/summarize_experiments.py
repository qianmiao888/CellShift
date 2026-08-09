from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "共享编码器，无批次分支": "ablation_monolithic_mse",
    "残差分解，无批次分支": "ablation_residual_mse",
    "残差分解+批次校准": "ablation_residual_batch_mse",
    "共享编码器+批次校准": "ablation_monolithic_batch_mse",
    "共享编码器+批次校准+多目标": "balanced_monolithic_batch_multiloss",
    "去PCA初始化": "ablation_random_decoder",
    "去unknown-aware类别dropout": "ablation_no_category_dropout",
}


def main() -> None:
    experiment_root = ROOT / "artifacts" / "experiments"
    rows: list[dict[str, object]] = []
    for label, directory in EXPERIMENTS.items():
        metrics = json.loads(
            (experiment_root / directory / "metrics.json").read_text(encoding="utf-8")
        )["metrics"]
        rows.append(
            {
                "method": label,
                "balanced_log2_rmse": sum(row["log2_rmse"] for row in metrics) / len(metrics),
                "balanced_delta_pcc": sum(row["delta_pcc"] for row in metrics) / len(metrics),
                "best_epoch": json.loads(
                    (experiment_root / directory / "metrics.json").read_text(encoding="utf-8")
                )["best_epoch"],
            }
        )

    hybrid = pd.read_csv(
        experiment_root / "balanced_monolithic_batch_multiloss" / "hybrid_metrics.csv"
    )
    for method, group in hybrid.groupby("method"):
        rows.append(
            {
                "method": method,
                "balanced_log2_rmse": group["log2_rmse"].mean(),
                "balanced_delta_pcc": group["delta_pcc"].mean(),
                "best_epoch": None,
            }
        )
    output = ROOT / "artifacts" / "experiments" / "experiment_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

