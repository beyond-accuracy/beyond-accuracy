#!/usr/bin/env python3
"""Write the Croissant metadata (croissant.json) for the released LOO score tables.

Usage:
    python tools/make_croissant.py --root /path/to/beyond-accuracy-loo-scores [--base-url https://.../]
With no --base-url, file URLs are relative to croissant.json (for local validation).
Requires: pip install mlcroissant
"""
import argparse
import hashlib
import json
from pathlib import Path

import mlcroissant as mlc

AGENT_FIELDS = [
    ('model', mlc.DataType.TEXT, 'Forecasting model: DeMo, ForecastMAE, Wayformer, AutoBot or MTR.'),
    ('scenario_id', mlc.DataType.TEXT, 'Scenario id of the source dataset.'),
    ('focal_agent_id', mlc.DataType.TEXT, 'Track id of the focal agent.'),
    ('slot', mlc.DataType.INTEGER, "Distance rank of the masked agent in UniTraj's 32-slot input (1 = nearest)."),
    ('agent_id', mlc.DataType.TEXT, 'Track id of the masked agent.'),
    ('delta_minADE6', mlc.DataType.FLOAT, 'Change in minADE6 (m) when the agent is masked.'),
    ('delta_minFDE6', mlc.DataType.FLOAT, 'Change in minFDE6 (m) when the agent is masked.'),
    ('delta_brier_minFDE6', mlc.DataType.FLOAT, 'Change in Brier-minFDE6 when the agent is masked.'),
]
SCENE_FIELDS = [
    ('model', mlc.DataType.TEXT, 'Forecasting model.'),
    ('scenario_id', mlc.DataType.TEXT, 'Scenario id of the source dataset.'),
    ('focal_agent_id', mlc.DataType.TEXT, 'Track id of the focal agent.'),
    ('n_agents', mlc.DataType.INTEGER, 'Number of surrounding agents listed for this record.'),
] + [(f'baseline_{m}', mlc.DataType.FLOAT, f'Full-scene {m}.') for m in ('minADE6', 'minFDE6', 'brier_minFDE6')] + [
    (f'{g}_delta_{m}', mlc.DataType.FLOAT, f'Change in {m} when the {g[:-1]} 5 agents are removed together (NaN if fewer agents).')
    for g in ('top5', 'random5', 'bottom5') for m in ('minADE6', 'minFDE6', 'brier_minFDE6')
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--base-url', default='')
    args = ap.parse_args()

    files, record_sets = [], []
    for ds, ds_name in (('av2', 'Argoverse 2 Motion Forecasting (validation)'), ('waymo', 'Waymo Open Motion Dataset v1.1 (validation, SDC focal)')):
        for table, fields, what in (('agent_scores', AGENT_FIELDS, 'one row per (model, scenario, focal agent, masked agent)'),
                                    ('scene_summary', SCENE_FIELDS, 'one row per (model, scenario, focal agent)')):
            rel = f'{ds}/{table}.parquet'
            fid = f'{ds}_{table}_file'
            files.append(mlc.FileObject(
                id=fid, name=f'{ds}_{table}.parquet', description=f'{table} for {ds_name}.',
                content_url=args.base_url + rel, encoding_formats=['application/x-parquet'],
                sha256=sha256(args.root / rel)))
            rs_id = f'{ds}_{table}'
            record_sets.append(mlc.RecordSet(
                id=rs_id, name=rs_id, description=f'{ds_name}: {what}.',
                fields=[mlc.Field(id=f'{rs_id}/{name}', name=name, description=desc, data_types=[dtype],
                                  source=mlc.Source(file_object=fid, extract=mlc.Extract(column=name)))
                        for name, dtype, desc in fields]))

    metadata = mlc.Metadata(
        name='beyond-accuracy-loo-scores',
        description=('Per-scene leave-one-out (LOO) agent-removal sensitivity scores of five motion forecasting models '
                     '(DeMo, ForecastMAE, Wayformer, AutoBot, MTR) on the Argoverse 2 and Waymo Open Motion validation sets. '
                     'Each row gives the change in minADE6, minFDE6 and Brier-minFDE6 against the recorded future when one '
                     'surrounding agent is masked. Scores measure output sensitivity to agent removal, not causal influence. '
                     'av2/ is CC BY-NC-SA 4.0 (derived from Argoverse 2, (c) 2021 Argo AI, LLC); waymo/ is released under the '
                     'Waymo Dataset License Agreement for Non-Commercial Use, and any further downstream use or modification '
                     'is subject to that Agreement.'),
        cite_as=('@inproceedings{park2026beyond, title={Beyond Accuracy: Auditing Agent-Removal Sensitivity in Motion '
                 'Forecasting}, author={Park, Geonyeong and Park, Byounghun and Kim, Nayoung and Kim, Kyungmin and Hwang, '
                 'Soonmin}, booktitle={The Fortieth Annual Conference on Neural Information Processing Systems Evaluations '
                 'and Datasets Track}, year={2026}, url={https://openreview.net/forum?id=7axaNwvSMD}}'),
        url='https://beyond-accuracy.github.io/',
        license=['https://creativecommons.org/licenses/by-nc-sa/4.0/', 'https://waymo.com/open/terms/'],
        version='1.0.0',
        date_published='2026-11-06',
        creators=[mlc.Person(name=n) for n in ('Geonyeong Park', 'Byounghun Park', 'Nayoung Kim', 'Kyungmin Kim', 'Soonmin Hwang')],
        distribution=files,
        record_sets=record_sets,
    )
    out = args.root / 'croissant.json'
    out.write_text(json.dumps(metadata.to_json(), indent=2, default=str), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
