# Beyond Accuracy: Auditing Agent-Removal Sensitivity in Motion Forecasting

Geonyeong Park, Byounghun Park, Nayoung Kim, Kyungmin Kim, Soonmin Hwang (Hanyang University)

NeurIPS 2026 Evaluations & Datasets Track ·
[Project page](https://beyond-accuracy.github.io/) ·
[Paper](https://openreview.net/forum?id=7axaNwvSMD) ·
[LOO scores](https://huggingface.co/datasets/madpyro/beyond-accuracy-loo-scores)

The LOO Reliance Audit masks each surrounding agent $a_i$ of a scene $S$ in turn and records how much the
forecast of the focal agent degrades against the recorded future $y$:

$$\Delta_i = \mathrm{minADE}_6\big(f(S \setminus \{a_i\}),\, y\big) - \mathrm{minADE}_6\big(f(S),\, y\big)$$

This repository contains the audit and the analyses behind the three findings, built on
[UniTraj](https://github.com/vita-epfl/UniTraj):

| | Script |
|---|---|
| LOO audit (per scene, per agent; minADE6 / minFDE6 / Brier-minFDE6) | `tools/experiments/loo_analysis.py`, `unitraj/analysis/loo_analyzer.py` |
| F1: agreement of LOO rankings across models | `tools/experiments/run_cross_model.py` |
| F2: alignment with Waymo CausalAgents labels (Prec@1) | `tools/parse_causal_labels.py`, `tools/experiments/causal_alignment.py` |
| F3: attention versus LOO | `tools/experiments/extract_attention.py`, `tools/experiments/analyze_attention_vs_loo.py` |

Five models: AutoBot, Wayformer and MTR (from UniTraj), and DeMo and ForecastMAE (ported into UniTraj).
Per-scene LOO scores are on [Hugging Face](https://huggingface.co/datasets/madpyro/beyond-accuracy-loo-scores)
(exported by `tools/export_loo_scores.py`, Croissant metadata by `tools/make_croissant.py`).
The trained checkpoints will be added on the project page.

## Installation

Tested with Python 3.10, PyTorch 2.1.1 (CUDA 11.8) and one NVIDIA RTX A6000 (48 GB) per job.

```bash
conda create -n beyond-accuracy python=3.10 -y && conda activate beyond-accuracy
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop            # builds the CUDA ops used by MTR
```

DeMo needs `mamba-ssm` and `causal-conv1d`, which compile against your CUDA toolkit.

## Data

1. Convert the Argoverse 2 and Waymo Open Motion (v1.1) validation sets to ScenarioNet format as described in
   [UniTraj](https://github.com/vita-epfl/UniTraj), and set the paths in `unitraj/configs/dataset/{av2,waymo}.yaml`.
2. Set `cache_path` in `unitraj/configs/config.yaml` to an absolute path.
3. Build the preprocessed cache once per dataset. All five models read the same cache, built with the DeMo
   settings, as in the paper:

```bash
python tools/build_cache.py method=DeMo dataset=av2 +val_only=True
python tools/build_cache.py method=DeMo dataset=waymo only_train_on_ego=True +val_only=True   # SDC as focal agent
```

Drop `+val_only=True` to also build the training cache. `av2.yaml` fixes the AV2 map preprocessing used for the paper's
AV2 models (100 m crop, every lane resampled at 0.5 m); Waymo uses the method defaults (150 m).

For F2, parse the Waymo CausalAgents labels (released with the Waymo Open Dataset). This step needs
TensorFlow and `waymo-open-dataset`, best in a separate environment:

```bash
python tools/parse_causal_labels.py --input /path/to/causal_labels.tfrecord --output unitraj/data/causal_labels_lookup.pkl
```

## Models and checkpoints

All models were trained with seed 42. Checkpoints go under `ckpts/` with these names. LOO time is one full
validation pass of the audit on one RTX A6000 (48 GB).

| Model | `method=` | AV2 checkpoint (epoch) | Waymo checkpoint (epoch) | LOO time, AV2 / Waymo |
|---|---|---|---|---|
| AutoBot | `autobot` | `ckpts/argoverse2/autobot_av2.ckpt` (79) | `ckpts/WOMD/Autobot_WOMD.ckpt` (90) | 3.3 h / 3.4 h |
| Wayformer | `wayformer` | `ckpts/argoverse2/wayformer_av2.ckpt` (118) | `ckpts/WOMD/Wayformer_WOMD.ckpt` (114) | 1.4 h / 2.8 h |
| MTR | `MTR` | `ckpts/argoverse2/MTR_av2.ckpt` (26) | `ckpts/WOMD/MTR_WOMD.ckpt` (38) | 4.3 h / 6.5 h |
| DeMo | `DeMo` | `ckpts/argoverse2/DeMo_av2.ckpt` (57) | `ckpts/WOMD/DeMo_WOMD.ckpt` (46) | 2.5 h / 4.6 h |
| ForecastMAE | `forecast` (pre-training: `MAE`) | `ckpts/argoverse2/forecast_mae_av2.ckpt` (58) | `ckpts/WOMD/Forecast_WOMD.ckpt` (78) | 2.7 h / 5.0 h |

To train a model yourself: `cd unitraj && python train.py method=DeMo dataset=av2`
(set `pretrained_weights` in `configs/method/forecast.yaml` to fine-tune ForecastMAE from its MAE pre-training).
`python evaluation.py method=DeMo dataset=av2 ckpt_path=...` reports minADE6, minFDE6, miss rate and Brier-minFDE6.

## Running the audit

One model on one dataset:

```bash
python tools/experiments/loo_analysis.py --model DeMo --dataset waymo --sdc-only \
    --ckpt ckpts/WOMD/DeMo_WOMD.ckpt --output results/loo/waymo/DeMo.pkl --seed 42
python tools/experiments/loo_analysis.py --model DeMo --dataset av2 \
    --ckpt ckpts/argoverse2/DeMo_av2.ckpt --output results/loo/av2/DeMo.pkl --seed 42
```

All five models on both datasets, two GPUs in parallel (about 10 h for AV2 and 15 h for Waymo on two RTX A6000):

```bash
bash scripts/run_loo_full_2datasets.sh
```

Each output `.pkl` maps `scenario_id → focal_agents → agent_id` to a record with `causal_vector` ($\Delta_i$ per agent slot,
minADE6), `causal_vectors` (minADE6 / minFDE6 / Brier-minFDE6), `skip_mask` (empty slots), `focal_idx` and the full-scene
errors `baseline_minADE6` etc. Waymo uses the SDC as the focal agent (44,058 scenes); AV2 uses the dataset's focal agent
(22,979 scenes).

## Analyses

```bash
# F1: pairwise Spearman rho, Kendall tau and Jaccard@5 between models (mean over scenes)
python tools/experiments/run_cross_model.py --loo-dir results/loo/waymo --output results/cross_model_waymo.pkl
python tools/experiments/run_cross_model.py --loo-dir results/loo/av2   --output results/cross_model_av2.pkl

# F2: LOO top-k against the human causal labels (Waymo)
python tools/experiments/causal_alignment.py --loo results/loo/waymo/*.pkl

# F3: attention versus LOO (DeMo, ForecastMAE, Wayformer, AutoBot; MTR has no dense agent-to-agent attention)
bash tools/experiments/run_all_attention.sh waymo
python tools/experiments/analyze_attention_vs_loo.py --dataset waymo
```

## Citation

```bibtex
@inproceedings{park2026beyond,
  title     = {Beyond Accuracy: Auditing Agent-Removal Sensitivity in Motion Forecasting},
  author    = {Park, Geonyeong and Park, Byounghun and Kim, Nayoung and Kim, Kyungmin and Hwang, Soonmin},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://openreview.net/forum?id=7axaNwvSMD}
}
```

## License and acknowledgements

This code builds on [UniTraj](https://github.com/vita-epfl/UniTraj) and is released under the GNU AGPLv3
(`LICENSE`, `docs/LICENSE.AGPL`), as UniTraj is. MTR follows [MTR](https://github.com/sshaoshuai/MTR) (Apache-2.0).
The DeMo and ForecastMAE models are ported from [DeMo](https://github.com/fudan-zvg/DeMo) and
[Forecast-MAE](https://github.com/jchengai/forecast-mae); please also cite those works when you use them.
