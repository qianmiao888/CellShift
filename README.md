# CellShift

**CellShift：面向条件响应预测的酵母虚拟细胞**

GOAI AI for Research 算法赛·虚拟细胞方向的封闭数据榜方案。任务是在不把真实
matched control 蛋白向量作为模型输入的条件下，根据菌株、化合物、培养条件和观测
过程，直接预测扰动后的 4,422 维 log2 蛋白质组。

最终方案由训练集内生的分解统计先验和 unknown-aware 共享条件编码器组成，并根据
OOD 场景确定性路由：新化合物与时间场景使用统计分支，新菌株与双重未知场景使用
神经分支。

## Current evidence

| Validation summary | Value |
|---|---:|
| Retained proteins | 4,422 |
| Frozen training rows | 5,920 |
| Scenario-router mean RMSE | 0.4298 |
| Scenario-router mean Δ PCC | 0.3925 |
| PCA residual variance explained (64D) | 82.03% |

These are internal diagnostics on the organizer-provided frozen validation
splits, not public-leaderboard scores. Full attribution and ablations are in
[`docs/CellShift_初赛方案.md`](docs/CellShift_初赛方案.md).

## Data boundary

- Training and model selection use only `input/*train_val*.csv`.
- `input/*raw_test*.csv` is never loaded by training or validation scripts.
- Protein filtering statistics are fitted on rows where `split_final == train`.
- Missing targets are retained as a mask and never contribute gradients or metrics.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python scripts/audit_data.py --input-dir input
python scripts/run_baselines.py --input-dir input
python scripts/train_model.py --input-dir input --experiment-name balanced_monolithic_batch_multiloss --architecture monolithic --use-batch-calibration --latent-dim 64 --hidden-dim 256 --epochs 50 --patience 10 --batch-size 128 --learning-rate 0.0003 --fc-weight 0.2 --protein-corr-weight 0.05
python scripts/evaluate_hybrid.py --input-dir input
python scripts/check_submission.py --input-dir input

$env:PYTHONPATH='src'
python -m unittest discover -s tests
```

Outputs are written under `artifacts/` and contain the exact configuration and
diagnostic tables needed for the preliminary-round submission document.

## Repository layout

- `src/virtual_cell/`: data contract, features, losses, metrics and models.
- `scripts/`: auditable training, evaluation and submission checks.
- `config/`: default experiment configuration.
- `artifacts/`: small configs, histories and diagnostic tables; no weights.
- `docs/`: preliminary-round technical proposal in Markdown.
- `tests/`: unknown-entity mapping and metric tests.

## License

Code is released under Apache-2.0. Competition data is not redistributed and
remains subject to the organizer's terms.
