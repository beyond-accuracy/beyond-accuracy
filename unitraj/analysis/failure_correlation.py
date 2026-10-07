"""Failure correlation analysis.

Computes per-scene disagreement score and correlates it with prediction error
to test whether high inter-model disagreement predicts hard scenes.
"""
import numpy as np
from scipy.stats import spearmanr


def compute_disagreement(jaccard_scores: dict) -> float:
    """1 - mean(Jaccard@k) across all model pairs.

    Args:
        jaccard_scores: {'modelA__modelB': float, ...}
    Returns:
        disagreement in [0, 1]
    """
    if not jaccard_scores:
        return float('nan')
    return float(1.0 - np.mean(list(jaccard_scores.values())))


def compute_failure_correlation(cross_pkl: dict, loo_pkls: dict) -> dict:
    """Spearman correlation between disagreement and mean_minADE6 across scenes.

    Args:
        cross_pkl: output of compute_cross_model_agreement (has 'per_scene')
        loo_pkls:  {model_name: pkl_dict} — same PKLs used for cross_model
    Returns:
        {
          'spearman_r':  float,
          'spearman_p':  float,
          'per_scene': {
            scenario_id: {
              agent_id: {
                'disagreement':  float,
                'mean_minADE6':  float | None,
              }
            }
          }
        }
    """
    model_names = list(loo_pkls.keys())
    per_scene_out: dict = {}

    disagreements = []
    mean_ades = []

    for sid, agents in cross_pkl['per_scene'].items():
        per_scene_out[sid] = {}

        for agent_id, metrics in agents.items():
            jaccard_scores = metrics.get('jaccard', {})
            disagreement = compute_disagreement(jaccard_scores)

            # mean_minADE6 across models — if ANY model returns None, result is None
            baselines = []
            has_none = False
            for m in model_names:
                try:
                    b = loo_pkls[m]['results'][sid]['focal_agents'][agent_id]['baseline_minADE6']
                    if b is None:
                        has_none = True
                        break
                    baselines.append(float(b))
                except KeyError:
                    pass

            mean_ade = None if has_none else (float(np.mean(baselines)) if baselines else None)

            per_scene_out[sid][agent_id] = {
                'disagreement': disagreement,
                'mean_minADE6': mean_ade,
            }

            # only include in global correlation if both values are valid
            if mean_ade is not None and not np.isnan(disagreement):
                disagreements.append(disagreement)
                mean_ades.append(mean_ade)

    # Spearman correlation
    if len(disagreements) >= 3:
        result = spearmanr(disagreements, mean_ades)
        spearman_r = float(result.statistic)
        spearman_p = float(result.pvalue)
    else:
        spearman_r = float('nan')
        spearman_p = float('nan')

    return {
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'per_scene':  per_scene_out,
    }
