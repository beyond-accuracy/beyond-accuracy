#!/usr/bin/env python3
"""Export LOO audit outputs (tools/experiments/loo_analysis.py .pkl files) to the released Parquet tables.

For each dataset it writes two tables:
  agent_scores.parquet   one row per (model, scenario, focal agent, surrounding agent):
                         Delta_i under minADE6 / minFDE6 / Brier-minFDE6
  scene_summary.parquet  one row per (model, scenario, focal agent): full-scene errors and the
                         top-5 / random-5 / bottom-5 group-removal deltas used for faithfulness

Usage:
    python tools/export_loo_scores.py --loo-dir results/loo/waymo --dataset waymo --out release/waymo
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {  # file stem -> name used in the paper
    'DeMo': 'DeMo', 'forecast': 'ForecastMAE', 'wayformer': 'Wayformer', 'autobot': 'AutoBot', 'MTR': 'MTR',
}
METRICS = {'minADE6': 'minADE6', 'minFDE6': 'minFDE6', 'brier_minFDE6': 'brier_minFDE6'}
GROUPS = {'top_k': 'top5', 'random_k': 'random5', 'bottom_k': 'bottom5'}
GROUP_METRICS = {'delta_ade': 'minADE6', 'delta_fde6': 'minFDE6', 'delta_brier': 'brier_minFDE6'}


def export_model(pkl_path: Path, model: str):
    results = pickle.load(open(pkl_path, 'rb'))['results']
    agent_cols = {k: [] for k in ('scenario_id', 'focal_agent_id', 'slot', 'agent_id',
                                  *(f'delta_{m}' for m in METRICS.values()))}
    scene_rows = []
    for sid, scene in results.items():
        for fid, rec in scene['focal_agents'].items():
            skip = np.asarray(rec['skip_mask'], dtype=bool)
            ids = rec['agent_view']['agent_ids']
            slots = np.flatnonzero(~skip)
            n = len(slots)
            agent_cols['scenario_id'] += [str(sid)] * n
            agent_cols['focal_agent_id'] += [str(fid)] * n
            agent_cols['slot'] += slots.tolist()
            agent_cols['agent_id'] += [str(ids[s]) for s in slots]
            for key, name in METRICS.items():
                agent_cols[f'delta_{name}'] += np.asarray(rec['causal_vectors'][key], dtype=np.float32)[slots].tolist()
            row = {'scenario_id': str(sid), 'focal_agent_id': str(fid), 'n_agents': n}
            for name in METRICS.values():
                row[f'baseline_{name}'] = float(rec[f'baseline_{name}'])
            faith = rec.get('faithfulness') or {}
            for gkey, gname in GROUPS.items():
                group = faith.get(gkey) or {}  # empty when the scene has too few agents for a group of 5
                for mkey, mname in GROUP_METRICS.items():
                    value = group.get(mkey)
                    row[f'{gname}_delta_{mname}'] = float(value) if value is not None else np.nan
            scene_rows.append(row)
    agents = pd.DataFrame(agent_cols)
    agents.insert(0, 'model', model)
    scenes = pd.DataFrame(scene_rows)
    scenes.insert(0, 'model', model)
    return agents, scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loo-dir', type=Path, required=True)
    ap.add_argument('--dataset', required=True, choices=['av2', 'waymo'])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    agents, scenes = [], []
    for stem, name in MODELS.items():
        a, s = export_model(args.loo_dir / f'{stem}.pkl', name)
        print(f'{args.dataset} {name}: {s.shape[0]} (scene, focal) records, {a.shape[0]} agent rows')
        agents.append(a)
        scenes.append(s)
    agents = pd.concat(agents, ignore_index=True)
    scenes = pd.concat(scenes, ignore_index=True)
    for df in (agents, scenes):
        df['model'] = df['model'].astype('category')
    agents['slot'] = agents['slot'].astype('int8')
    scenes['n_agents'] = scenes['n_agents'].astype('int16')
    agents.to_parquet(args.out / 'agent_scores.parquet', compression='zstd', index=False)
    scenes.to_parquet(args.out / 'scene_summary.parquet', compression='zstd', index=False)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
