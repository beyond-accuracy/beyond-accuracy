#!/usr/bin/env python3
"""Evaluate alignment between LOO reliance rankings and Waymo CausalAgents GT.

Prerequisites:
  1. unitraj/data/causal_labels_lookup.pkl (created by tools/parse_causal_labels.py)
  2. At least one SDC-centric LOO result pkl
     (generated with loo_analysis.py --sdc-only)

Metrics:
  - Top-1 accuracy  : LOO top-1 agent ∈ causal GT (union)
  - Recall@k        : |LOO_top_k ∩ causal_GT| / |causal_GT|
  - Precision@k     : |LOO_top_k ∩ causal_GT| / k
  - AUPRC           : area under precision-recall curve (varying k)

Usage:
    python tools/experiments/causal_alignment.py \
        --loo results/loo/waymo_sdc/DeMo.pkl results/loo/waymo_sdc/MTR.pkl \
        --causal-labels unitraj/data/causal_labels_lookup.pkl \
        [--gt-mode union|majority]   # default: union
        [--output results/causal_alignment.csv]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _sorted_agents(agent_view: dict, skip_empty: bool = True) -> list[str]:
    """Return agent_ids sorted by causal_vector, descending (skipped excluded)."""
    scores    = agent_view['scores']      # [N]
    skip_mask = agent_view['skip_mask']   # [N] bool
    agent_ids = agent_view['agent_ids']   # [N] str

    ranked = []
    for i in np.argsort(scores)[::-1]:
        if skip_mask[i]:
            continue
        aid = str(agent_ids[i])
        if skip_empty and aid == '':
            continue
        ranked.append(aid)
    return ranked


def top1_acc(ranked: list[str], causal_gt: set[str]) -> float | None:
    """1.0 if the LOO top-1 agent is in causal_GT, else 0.0. None if ranked is empty."""
    if not ranked:
        return None
    return float(ranked[0] in causal_gt)


def recall_at_k(ranked: list[str], causal_gt: set[str], k: int) -> float | None:
    """Recall@k: |top_k ∩ GT| / |GT|. None if GT is empty."""
    if not causal_gt:
        return None
    top_k = set(ranked[:k])
    return len(top_k & causal_gt) / len(causal_gt)


def precision_at_k(ranked: list[str], causal_gt: set[str], k: int) -> float | None:
    """Precision@k: |top_k ∩ GT| / k. If len(ranked) < k, divides by the number returned."""
    if not ranked:
        return None
    top_k = ranked[:k]
    if not top_k:
        return None
    return sum(1 for a in top_k if a in causal_gt) / len(top_k)


def auprc(ranked: list[str], causal_gt: set[str]) -> float | None:
    """AUPRC: area under the PR curve, varying the threshold over k=1..len(ranked).

    interpolation-free version (standard sklearn-style).
    None if GT or ranked is empty.
    """
    if not causal_gt or not ranked:
        return None

    precisions, recalls = [], []
    hits = 0
    for k, aid in enumerate(ranked, start=1):
        if aid in causal_gt:
            hits += 1
        precisions.append(hits / k)
        recalls.append(hits / len(causal_gt))

    if recalls[-1] == 0:
        return 0.0

    # trapezoidal integration
    area = 0.0
    for i in range(1, len(recalls)):
        dr = recalls[i] - recalls[i - 1]
        area += dr * (precisions[i] + precisions[i - 1]) / 2
    return float(area)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(loo_payload: dict, causal_lookup: dict, gt_mode: str, ks: list[int]) -> dict:
    """Evaluate all scenes/focal agents in one model's LOO pkl.

    Returns:
        {
          'top1': list[float],
          'recall_k': {k: list[float]},
          'precision_k': {k: list[float]},
          'auprc': list[float],
          'n_scenes': int,
          'n_matched': int,   # number of scenes with causal GT
        }
    """
    results_all = loo_payload['results']  # {scenario_id → result_dict}

    top1_list:    list[float] = []
    recall_k_list = {k: [] for k in ks}
    prec_k_list   = {k: [] for k in ks}
    auprc_list:   list[float] = []
    n_matched = 0

    for scenario_id, scene_result in results_all.items():
        # causal GT lookup
        gt_entry = causal_lookup.get(str(scenario_id))
        if gt_entry is None:
            continue
        causal_gt: set[str] = gt_entry[gt_mode]
        if not causal_gt:
            continue
        n_matched += 1

        focal_agents = scene_result.get('focal_agents', {})
        for _agent_id, fa in focal_agents.items():
            agent_view = fa.get('agent_view')
            if agent_view is None:
                continue

            ranked = _sorted_agents(agent_view)
            if not ranked:
                continue

            v = top1_acc(ranked, causal_gt)
            if v is not None:
                top1_list.append(v)

            for k in ks:
                r = recall_at_k(ranked, causal_gt, k)
                p = precision_at_k(ranked, causal_gt, k)
                if r is not None:
                    recall_k_list[k].append(r)
                if p is not None:
                    prec_k_list[k].append(p)

            a = auprc(ranked, causal_gt)
            if a is not None:
                auprc_list.append(a)

    return {
        'top1':        top1_list,
        'recall_k':    recall_k_list,
        'precision_k': prec_k_list,
        'auprc':       auprc_list,
        'n_scenes':    len(results_all),
        'n_matched':   n_matched,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='LOO ↔ CausalAgents GT alignment evaluation')
    parser.add_argument('--loo', nargs='+', required=True, help='SDC LOO pkl files')
    parser.add_argument('--causal-labels', default='unitraj/data/causal_labels_lookup.pkl')
    parser.add_argument('--gt-mode', choices=['union', 'majority'], default='union',
                        help='causal GT set definition: union(≥1) or majority(≥3/5)')
    parser.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5],
                        help='k values for Recall@k / Precision@k')
    parser.add_argument('--output', default=None, help='Output CSV path (optional)')
    args = parser.parse_args()

    print(f'Loading causal labels: {args.causal_labels}')
    with open(args.causal_labels, 'rb') as f:
        causal_lookup = pickle.load(f)
    print(f'  {len(causal_lookup)} scenarios in lookup\n')

    ks = sorted(args.ks)
    rows = []

    for loo_path in args.loo:
        print(f'Evaluating: {loo_path}')
        with open(loo_path, 'rb') as f:
            payload = pickle.load(f)

        model_name = payload.get('model', Path(loo_path).stem)
        metrics = evaluate_model(payload, causal_lookup, gt_mode=args.gt_mode, ks=ks)

        n_s = metrics['n_scenes']
        n_m = metrics['n_matched']
        top1 = np.mean(metrics['top1']) if metrics['top1'] else float('nan')
        aupr = np.mean(metrics['auprc']) if metrics['auprc'] else float('nan')

        print(f'  Model       : {model_name}')
        print(f'  Scenes      : {n_s} total, {n_m} matched (causal GT exists)')
        print(f'  Top-1 acc   : {top1:.4f}')
        for k in ks:
            r = np.mean(metrics['recall_k'][k])    if metrics['recall_k'][k]    else float('nan')
            p = np.mean(metrics['precision_k'][k]) if metrics['precision_k'][k] else float('nan')
            print(f'  Recall@{k}    : {r:.4f}   Precision@{k}: {p:.4f}')
        print(f'  AUPRC       : {aupr:.4f}')
        print()

        row: dict = {'model': model_name, 'n_scenes': n_s, 'n_matched': n_m, 'top1': top1, 'auprc': aupr}
        for k in ks:
            row[f'recall@{k}']    = np.mean(metrics['recall_k'][k])    if metrics['recall_k'][k]    else float('nan')
            row[f'precision@{k}'] = np.mean(metrics['precision_k'][k]) if metrics['precision_k'][k] else float('nan')
        rows.append(row)

    if len(rows) > 1:
        print('=== Summary Table ===')
        header = ['model', 'n_matched', 'top1', 'auprc'] + [f'recall@{k}' for k in ks] + [f'precision@{k}' for k in ks]
        print('\t'.join(header))
        for row in rows:
            vals = [str(row.get(h, '')) if isinstance(row.get(h), str)
                    else f'{row.get(h, float("nan")):.4f}' for h in header]
            vals[0] = row['model']
            vals[1] = str(row['n_matched'])
            print('\t'.join(vals))

    if args.output:
        import csv
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['model', 'n_scenes', 'n_matched', 'top1', 'auprc'] + \
                     [f'recall@{k}' for k in ks] + [f'precision@{k}' for k in ks]
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
