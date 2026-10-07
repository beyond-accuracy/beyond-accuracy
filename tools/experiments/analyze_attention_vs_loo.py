#!/usr/bin/env python3
"""Attention vs LOO: joint three-axis analysis.

Runs four analyses:
  A. Attention vs LOO (same model, per-scene Spearman r)
     → Does the model's internal attention match its actual reliance (LOO)?
  B. Attention vs Causal GT (Waymo only, Precision@1 + MRR)
     → Does attention also find human-labeled causal agents? Compared with LOO.
  C. Cross-model attention divergence (pairwise Spearman r)
     → Does attention also give ρ ≈ 0.02?
  D. Three-axis triangle (Waymo only)
     → Agreement among (Attention top-1, LOO top-1, Causal GT).

Usage:
    python tools/experiments/analyze_attention_vs_loo.py --dataset waymo
    python tools/experiments/analyze_attention_vs_loo.py --dataset av2
"""
import argparse
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm


MODELS = ['DeMo', 'forecast', 'wayformer', 'autobot']

# Model name → LOO pkl filename (stem)
LOO_FILENAME = {
    'waymo': {'DeMo': 'DeMo', 'forecast': 'forecast', 'wayformer': 'wayformer', 'autobot': 'autobot'},
    'av2':   {'DeMo': 'DeMo', 'forecast': 'forecast', 'wayformer': 'wayformer', 'autobot': 'autobot'},
}


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_attention(dataset: str) -> dict:
    """Load 4-model attention pkls. Returns {model: {sid: (attn_matrix, agent_ids)}}."""
    out = {}
    for m in MODELS:
        with open(f'results/attention/{dataset}/{m}.pkl', 'rb') as f:
            d = pickle.load(f)
        out[m] = {
            sid: (
                sc['agent_attention'].astype(np.float32),
                list(sc['agent_ids']),
            )
            for sid, sc in d['results'].items()
        }
        print(f'  [attention] {m}: {len(out[m])} scenes')
    return out


def load_loo(dataset: str) -> dict:
    """Load 4-model LOO pkls. Returns {model: {sid: (scores, skip_mask, focal_idx, agent_ids)}}."""
    dir_map = {'waymo': 'results/loo/waymo', 'av2': 'results/loo/av2'}
    loo_dir = dir_map[dataset]
    name_map = LOO_FILENAME[dataset]
    out = {}
    for m in MODELS:
        loo_stem = name_map[m]
        with open(f'{loo_dir}/{loo_stem}.pkl', 'rb') as f:
            d = pickle.load(f)
        per_scene = {}
        for sid, scene in d['results'].items():
            fa_dict = scene.get('focal_agents', {})
            if not fa_dict:
                continue
            # Use first focal agent (SDC for Waymo, first focal for AV2)
            fa = next(iter(fa_dict.values()))
            scores = np.asarray(fa['causal_vector'], dtype=np.float32)
            skip = np.asarray(fa['skip_mask'], dtype=bool)
            focal_idx = int(fa.get('focal_idx', 0))
            # agent_ids (Waymo only; AV2 uses slot index)
            if 'agent_view' in fa and 'agent_ids' in fa['agent_view']:
                aids = list(fa['agent_view']['agent_ids'])
            else:
                aids = [str(i) for i in range(len(scores))]
            per_scene[sid] = (scores, skip, focal_idx, aids)
        out[m] = per_scene
        print(f'  [LOO]       {m}: {len(out[m])} scenes')
    return out


def load_causal_gt(path: str) -> dict:
    """Load Waymo causal labels lookup. Returns {sid: set of agent_id strings}."""
    with open(path, 'rb') as f:
        lookup = pickle.load(f)
    # Use majority (≥3/5 annotators)
    out = {}
    for sid, entry in lookup.items():
        gt = entry.get('majority', set())
        if gt:
            out[str(sid)] = gt
    print(f'  [Causal GT] {len(out)} scenes with majority causal agents')
    return out


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def attention_to_importance(attn: np.ndarray, focal_idx: int, skip: np.ndarray) -> np.ndarray:
    """Convert (N, N) attention matrix → (N,) agent importance vector.

    Uses the focal agent's row: how much focal agent attends to each other agent.
    Self (focal_idx) and skipped agents are masked to -inf.
    """
    N = attn.shape[0]
    if focal_idx < 0 or focal_idx >= N:
        vec = attn.mean(axis=0).astype(np.float32)  # fallback: mean over rows
    else:
        vec = attn[focal_idx].astype(np.float32).copy()
    # Mask
    vec[focal_idx] = -np.inf
    if skip is not None and skip.shape[0] == N:
        vec[skip] = -np.inf
    return vec


def valid_mask(skip: np.ndarray, focal_idx: int, N: int) -> np.ndarray:
    """Return boolean mask of valid (non-skipped, non-focal) agents."""
    valid = np.ones(N, dtype=bool)
    if skip is not None:
        valid &= ~skip
    if 0 <= focal_idx < N:
        valid[focal_idx] = False
    return valid


def top1_agent_id(vec: np.ndarray, agent_ids: list[str], valid: np.ndarray) -> str | None:
    """Return agent_id of argmax over valid indices. None if no valid."""
    if not valid.any():
        return None
    masked = np.where(valid, vec, -np.inf)
    idx = int(np.argmax(masked))
    if 0 <= idx < len(agent_ids):
        return str(agent_ids[idx])
    return None


def mrr_from_ranking(ranked_ids: list[str], causal_gt: set[str]) -> float:
    for i, aid in enumerate(ranked_ids):
        if aid in causal_gt:
            return 1.0 / (i + 1)
    return 0.0


def sorted_agent_ids_by_vec(vec: np.ndarray, agent_ids: list[str], valid: np.ndarray) -> list[str]:
    """Agent IDs sorted by vec descending, excluding invalid."""
    out = []
    masked = np.where(valid, vec, -np.inf)
    order = np.argsort(masked)[::-1]
    for i in order:
        if not valid[i]:
            continue
        if 0 <= i < len(agent_ids):
            out.append(str(agent_ids[i]))
    return out


# ─────────────────────────────────────────────────────────────────────
# Analysis A: Attention vs LOO per-scene Spearman r
# ─────────────────────────────────────────────────────────────────────

def analysis_A(attn_data: dict, loo_data: dict, common_scenes: set) -> dict:
    print('\n' + '=' * 70)
    print('Analysis A: Attention vs LOO (same model, per-scene Spearman r)')
    print('=' * 70)

    results = {}
    for m in MODELS:
        per_scene_r = []
        skipped = 0
        for sid in common_scenes:
            if sid not in attn_data[m] or sid not in loo_data[m]:
                skipped += 1
                continue
            attn_mat, attn_ids = attn_data[m][sid]
            loo_scores, skip, focal_idx, loo_ids = loo_data[m][sid]

            N = min(attn_mat.shape[0], len(loo_scores))
            attn_vec = attention_to_importance(attn_mat[:N, :N], focal_idx, skip[:N])
            valid = valid_mask(skip[:N], focal_idx, N)

            if valid.sum() < 3:
                skipped += 1
                continue

            r, _ = spearmanr(attn_vec[valid], loo_scores[:N][valid])
            if not np.isnan(r):
                per_scene_r.append(r)

        per_scene_r = np.array(per_scene_r)
        mean_r = per_scene_r.mean() if len(per_scene_r) > 0 else float('nan')
        median_r = np.median(per_scene_r) if len(per_scene_r) > 0 else float('nan')
        results[m] = {
            'n_scenes': len(per_scene_r),
            'mean_r': mean_r,
            'median_r': median_r,
            'std_r': per_scene_r.std() if len(per_scene_r) > 0 else 0.0,
        }
        print(f'  {m:<12}  n={len(per_scene_r):6d}  mean ρ={mean_r:+.4f}  median ρ={median_r:+.4f}  std={per_scene_r.std():.4f}')

    return results


# ─────────────────────────────────────────────────────────────────────
# Analysis B: Attention vs Causal GT (Waymo only)
# ─────────────────────────────────────────────────────────────────────

def analysis_B(attn_data: dict, loo_data: dict, causal_gt: dict, common_scenes: set) -> dict:
    print('\n' + '=' * 70)
    print('Analysis B: Attention/LOO vs Causal GT (Waymo majority)')
    print('=' * 70)

    results = {}
    scenes_with_gt = [s for s in common_scenes if s in causal_gt]
    print(f'  Scenes with causal GT: {len(scenes_with_gt)}')

    # Random baseline: for each scene, n_causal / n_valid
    baselines = []
    for sid in scenes_with_gt:
        fa = next(iter(loo_data['DeMo'][sid][1:2]))  # get skip_mask of DeMo as proxy
        loo_scores, skip, focal_idx, _ = loo_data['DeMo'][sid]
        n_valid = int((~skip).sum() - (1 if 0 <= focal_idx < len(skip) and not skip[focal_idx] else 0))
        n_causal = len(causal_gt[sid])
        if n_valid > 0:
            baselines.append(min(n_causal, n_valid) / n_valid)
    random_p1 = np.mean(baselines) if baselines else float('nan')
    print(f'  Random Prec@1 baseline: {random_p1:.4f}')

    for m in MODELS:
        n_hit_attn, n_hit_loo = 0, 0
        mrr_attn, mrr_loo = [], []
        n_total = 0

        for sid in scenes_with_gt:
            if sid not in attn_data[m] or sid not in loo_data[m]:
                continue
            gt = causal_gt[sid]

            attn_mat, attn_ids = attn_data[m][sid]
            loo_scores, skip, focal_idx, loo_ids = loo_data[m][sid]

            N = min(attn_mat.shape[0], len(loo_scores))
            attn_vec = attention_to_importance(attn_mat[:N, :N], focal_idx, skip[:N])
            valid = valid_mask(skip[:N], focal_idx, N)
            if valid.sum() == 0:
                continue

            # Use LOO's agent_ids for both (consistent)
            ids = loo_ids[:N] if len(loo_ids) >= N else [str(i) for i in range(N)]

            attn_ranked = sorted_agent_ids_by_vec(attn_vec, ids, valid)
            loo_ranked = sorted_agent_ids_by_vec(loo_scores[:N], ids, valid)

            if not attn_ranked or not loo_ranked:
                continue

            n_total += 1
            if attn_ranked[0] in gt:
                n_hit_attn += 1
            if loo_ranked[0] in gt:
                n_hit_loo += 1

            mrr_attn.append(mrr_from_ranking(attn_ranked, gt))
            mrr_loo.append(mrr_from_ranking(loo_ranked, gt))

        p1_attn = n_hit_attn / n_total if n_total > 0 else float('nan')
        p1_loo = n_hit_loo / n_total if n_total > 0 else float('nan')
        mrr_a = np.mean(mrr_attn) if mrr_attn else float('nan')
        mrr_l = np.mean(mrr_loo) if mrr_loo else float('nan')

        results[m] = {
            'n_total': n_total,
            'attn_p1': p1_attn,
            'loo_p1': p1_loo,
            'attn_mrr': mrr_a,
            'loo_mrr': mrr_l,
            'attn_lift': p1_attn / random_p1 if random_p1 else float('nan'),
            'loo_lift': p1_loo / random_p1 if random_p1 else float('nan'),
        }

    print(f'\n  {"Model":<12} {"n":>7} {"Attn P@1":>10} {"LOO P@1":>10} {"Attn MRR":>10} {"LOO MRR":>10}  {"Attn/rand":>10} {"LOO/rand":>10}')
    for m in MODELS:
        r = results[m]
        print(f'  {m:<12} {r["n_total"]:>7} {r["attn_p1"]:>10.4f} {r["loo_p1"]:>10.4f} '
              f'{r["attn_mrr"]:>10.4f} {r["loo_mrr"]:>10.4f}  '
              f'{r["attn_lift"]:>10.2f}x {r["loo_lift"]:>9.2f}x')

    results['_random_p1'] = random_p1
    return results


# ─────────────────────────────────────────────────────────────────────
# Analysis C: Cross-model attention divergence
# ─────────────────────────────────────────────────────────────────────

def analysis_C(attn_data: dict, loo_data: dict, common_scenes: set) -> dict:
    print('\n' + '=' * 70)
    print('Analysis C: Cross-model attention divergence')
    print('=' * 70)

    pair_results = {}
    for ma, mb in combinations(MODELS, 2):
        per_scene_r = []
        for sid in common_scenes:
            if sid not in attn_data[ma] or sid not in attn_data[mb]:
                continue
            # We need focal_idx and skip from any LOO (use ma)
            if sid not in loo_data[ma]:
                continue

            attn_a, _ = attn_data[ma][sid]
            attn_b, _ = attn_data[mb][sid]
            _, skip, focal_idx, _ = loo_data[ma][sid]

            N = min(attn_a.shape[0], attn_b.shape[0], len(skip))
            vec_a = attention_to_importance(attn_a[:N, :N], focal_idx, skip[:N])
            vec_b = attention_to_importance(attn_b[:N, :N], focal_idx, skip[:N])
            valid = valid_mask(skip[:N], focal_idx, N)
            if valid.sum() < 3:
                continue

            r, _ = spearmanr(vec_a[valid], vec_b[valid])
            if not np.isnan(r):
                per_scene_r.append(r)

        if per_scene_r:
            pair_results[(ma, mb)] = {
                'n': len(per_scene_r),
                'mean_r': float(np.mean(per_scene_r)),
                'std_r': float(np.std(per_scene_r)),
            }

    print(f'\n  {"Pair":<28} {"n":>8} {"mean ρ":>10} {"std":>10}')
    for (ma, mb), v in pair_results.items():
        print(f'  {ma+" — "+mb:<28} {v["n"]:>8} {v["mean_r"]:>10.4f} {v["std_r"]:>10.4f}')

    all_r = [v['mean_r'] for v in pair_results.values()]
    overall_mean = float(np.mean(all_r)) if all_r else float('nan')
    print(f'\n  Attention cross-model mean ρ = {overall_mean:+.4f}')
    print(f'  (LOO cross-model mean ρ reported in paper: ~0.02)')

    return {'pairs': pair_results, 'overall_mean': overall_mean}


# ─────────────────────────────────────────────────────────────────────
# Analysis D: 3-axis triangle (Waymo only)
# ─────────────────────────────────────────────────────────────────────

def analysis_D(attn_data: dict, loo_data: dict, causal_gt: dict, common_scenes: set) -> dict:
    print('\n' + '=' * 70)
    print('Analysis D: 3-axis triangle (Attention / LOO / Causal GT)')
    print('=' * 70)

    scenes_with_gt = [s for s in common_scenes if s in causal_gt]
    results = {}

    # Categories per (scene, model):
    #   AA_LL_GG: attention=LOO (same top-1) AND attention matches GT
    #     → "model is self-consistent AND correct"
    #   AA_LL_GG_no: attention=LOO AND both wrong
    #     → "self-consistent but wrong (shortcut?)"
    #   AL_disagree_A_correct: attention ≠ LOO; attention matches GT
    #   AL_disagree_L_correct: attention ≠ LOO; LOO matches GT
    #   AL_disagree_both_wrong: attention ≠ LOO; neither matches GT

    for m in MODELS:
        cats = {
            'self_consistent_correct': 0,
            'self_consistent_wrong': 0,
            'disagree_attn_correct': 0,
            'disagree_loo_correct': 0,
            'disagree_both_wrong': 0,
        }
        n_total = 0

        for sid in scenes_with_gt:
            if sid not in attn_data[m] or sid not in loo_data[m]:
                continue
            gt = causal_gt[sid]

            attn_mat, _ = attn_data[m][sid]
            loo_scores, skip, focal_idx, loo_ids = loo_data[m][sid]

            N = min(attn_mat.shape[0], len(loo_scores))
            attn_vec = attention_to_importance(attn_mat[:N, :N], focal_idx, skip[:N])
            valid = valid_mask(skip[:N], focal_idx, N)
            if valid.sum() == 0:
                continue

            ids = loo_ids[:N] if len(loo_ids) >= N else [str(i) for i in range(N)]

            attn_top1 = top1_agent_id(attn_vec, ids, valid)
            loo_top1 = top1_agent_id(loo_scores[:N], ids, valid)

            if attn_top1 is None or loo_top1 is None:
                continue

            n_total += 1
            agree = (attn_top1 == loo_top1)
            attn_correct = attn_top1 in gt
            loo_correct = loo_top1 in gt

            if agree:
                if attn_correct:
                    cats['self_consistent_correct'] += 1
                else:
                    cats['self_consistent_wrong'] += 1
            else:
                if attn_correct and not loo_correct:
                    cats['disagree_attn_correct'] += 1
                elif loo_correct and not attn_correct:
                    cats['disagree_loo_correct'] += 1
                else:
                    cats['disagree_both_wrong'] += 1

        results[m] = {'n_total': n_total, **cats}

    # Print table
    print(f'\n  {"Model":<12} {"n":>7} {"SC_✓":>8} {"SC_✗":>8} {"A_✓_only":>10} {"L_✓_only":>10} {"both_✗":>10} {"agree_rate":>12}')
    for m in MODELS:
        r = results[m]
        n = r['n_total']
        if n == 0:
            continue
        agree_rate = (r['self_consistent_correct'] + r['self_consistent_wrong']) / n
        print(f'  {m:<12} {n:>7} '
              f'{r["self_consistent_correct"]/n:>8.3f} '
              f'{r["self_consistent_wrong"]/n:>8.3f} '
              f'{r["disagree_attn_correct"]/n:>10.3f} '
              f'{r["disagree_loo_correct"]/n:>10.3f} '
              f'{r["disagree_both_wrong"]/n:>10.3f} '
              f'{agree_rate:>12.3f}')

    print('\n  Legend:')
    print('    SC_✓       = attention & LOO agree, both match causal GT')
    print('    SC_✗       = attention & LOO agree, neither matches GT')
    print('    A_✓_only   = attention ≠ LOO; only attention matches GT')
    print('    L_✓_only   = attention ≠ LOO; only LOO matches GT')
    print('    both_✗     = attention ≠ LOO; neither matches GT')
    print('    agree_rate = fraction of scenes where attention top-1 == LOO top-1')

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['waymo', 'av2'], required=True)
    parser.add_argument('--causal-labels', default='unitraj/data/causal_labels_lookup.pkl')
    parser.add_argument('--output', default=None, help='Save results to pkl (optional)')
    args = parser.parse_args()

    print(f'=== Attention vs LOO Analysis ({args.dataset}) ===\n')

    print('Loading attention pkls...')
    attn_data = load_attention(args.dataset)

    print('\nLoading LOO pkls...')
    loo_data = load_loo(args.dataset)

    # Common scenes across all models
    common = None
    for m in MODELS:
        s = set(attn_data[m].keys()) & set(loo_data[m].keys())
        common = s if common is None else (common & s)
    print(f'\nCommon scenes (attention ∩ LOO, all 4 models): {len(common)}')

    # Run analyses
    res_A = analysis_A(attn_data, loo_data, common)

    res_B, res_D = None, None
    if args.dataset == 'waymo':
        print('\nLoading causal GT...')
        causal = load_causal_gt(args.causal_labels)
        res_B = analysis_B(attn_data, loo_data, causal, common)

    res_C = analysis_C(attn_data, loo_data, common)

    if args.dataset == 'waymo':
        res_D = analysis_D(attn_data, loo_data, causal, common)

    if args.output:
        output = {
            'dataset': args.dataset,
            'n_common_scenes': len(common),
            'analysis_A': res_A,
            'analysis_B': res_B,
            'analysis_C': res_C,
            'analysis_D': res_D,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump(output, f)
        print(f'\nResults saved → {args.output}')

    print('\n=== DONE ===')


if __name__ == '__main__':
    main()
