"""Cross-model causal agreement analysis.

Computes Jaccard@k, Spearman, and Kendall tau between model pairs
based on per-scene causal_vector rankings.
"""
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr, kendalltau


def compute_jaccard_at_k(vec_a: np.ndarray, vec_b: np.ndarray, k: int) -> float:
    """Jaccard similarity between top-k agent sets.

    Args:
        vec_a, vec_b: causal_vector [N]
        k: number of top agents to compare
    Returns:
        |TopK_A ∩ TopK_B| / |TopK_A ∪ TopK_B|
    """
    top_a = set(np.argsort(vec_a)[::-1][:k])
    top_b = set(np.argsort(vec_b)[::-1][:k])
    union = top_a | top_b
    if not union:
        return 1.0
    return len(top_a & top_b) / len(union)


def compute_valid_rank_metrics(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    skip_a: np.ndarray,
    skip_b: np.ndarray,
) -> dict:
    """Rank correlation over agents valid in both models.

    Args:
        vec_a, vec_b: causal_vector [N]
        skip_a, skip_b: skip mask [N] (True = skip)
    Returns:
        {'spearman': float, 'kendall': float}  (NaN if fewer than 3 valid agents)
    """
    combined_skip = skip_a | skip_b
    valid = ~combined_skip
    if valid.sum() < 3:
        return {'spearman': float('nan'), 'kendall': float('nan')}
    a = vec_a[valid]
    b = vec_b[valid]
    if np.std(a) == 0 or np.std(b) == 0:
        return {'spearman': float('nan'), 'kendall': float('nan')}
    sp = spearmanr(a, b).statistic
    kt = kendalltau(a, b).statistic
    return {'spearman': float(sp), 'kendall': float(kt)}


def _extract_view_payload(focal_payload: dict, view: str | None, score_key: str):
    """Extract (scores, skip_mask) from focal agent data.

    If view is None, use the legacy causal_vector/skip_mask.
    Otherwise use focal_payload[view][score_key] and focal_payload[view]['skip_mask'].
    """
    if view is None:
        return (
            np.asarray(focal_payload['causal_vector'], dtype=float),
            np.asarray(focal_payload['skip_mask'], dtype=bool),
        )
    node = focal_payload[view]
    return (
        np.asarray(node[score_key], dtype=float),
        np.asarray(node['skip_mask'], dtype=bool),
    )


def compute_cross_model_agreement(loo_pkls: dict, k: int = 5,
                                   view: str | None = None,
                                   score_key: str = 'causal_vector') -> dict:
    """Cross-model causal agreement analysis.

    Args:
        loo_pkls: {model_name: pkl_dict} — each PKL in loo_analysis.py output format
        k: top-k size for Jaccard
    Returns:
        {
          'jaccard':  {'modelA__modelB': float (mean over scenes)},
          'spearman': {'modelA__modelB': float},
          'kendall':  {'modelA__modelB': float},
          'per_scene': {scenario_id: {agent_id: {'jaccard': {...}, 'spearman': {...}, 'kendall': {...}}}}
        }
    """
    model_names = list(loo_pkls.keys())
    pairs = list(combinations(model_names, 2))

    per_scene: dict = {}
    pair_scores: dict = {
        'jaccard':  {f'{a}__{b}': [] for a, b in pairs},
        'spearman': {f'{a}__{b}': [] for a, b in pairs},
        'kendall':  {f'{a}__{b}': [] for a, b in pairs},
    }

    # common scenes across all models
    common_scenes = set.intersection(*[
        set(pkl['results'].keys()) for pkl in loo_pkls.values()
    ])

    for sid in sorted(common_scenes):
        per_scene[sid] = {}
        # common focal agents across all models for this scene
        common_agents = set.intersection(*[
            set(loo_pkls[m]['results'][sid]['focal_agents'].keys())
            for m in model_names
        ])

        for agent_id in common_agents:
            per_scene[sid][agent_id] = {'jaccard': {}, 'spearman': {}, 'kendall': {}}

            for m_a, m_b in pairs:
                key = f'{m_a}__{m_b}'
                fa = loo_pkls[m_a]['results'][sid]['focal_agents'][agent_id]
                fb = loo_pkls[m_b]['results'][sid]['focal_agents'][agent_id]

                vec_a, skip_a = _extract_view_payload(fa, view, score_key)
                vec_b, skip_b = _extract_view_payload(fb, view, score_key)

                jac = compute_jaccard_at_k(vec_a, vec_b, k)
                rank = compute_valid_rank_metrics(vec_a, vec_b, skip_a, skip_b)

                per_scene[sid][agent_id]['jaccard'][key]  = jac
                per_scene[sid][agent_id]['spearman'][key] = rank['spearman']
                per_scene[sid][agent_id]['kendall'][key]  = rank['kendall']

                pair_scores['jaccard'][key].append(jac)
                pair_scores['spearman'][key].append(rank['spearman'])
                pair_scores['kendall'][key].append(rank['kendall'])

    # aggregate: nanmean over scenes
    def _nanmean(vals):
        arr = np.array(vals, dtype=float)
        return float(np.nanmean(arr)) if len(arr) > 0 else float('nan')

    return {
        'jaccard':   {k_: _nanmean(v) for k_, v in pair_scores['jaccard'].items()},
        'spearman':  {k_: _nanmean(v) for k_, v in pair_scores['spearman'].items()},
        'kendall':   {k_: _nanmean(v) for k_, v in pair_scores['kendall'].items()},
        'per_scene': per_scene,
    }
