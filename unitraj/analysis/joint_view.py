import math
import numpy as np


def build_joint_view(agent_scores, agent_skip, map_scores, map_skip, corridor_meta):
    raw_scores = []
    normalized_scores = []
    skip_mask = []
    unit_meta = []

    for idx, score in enumerate(agent_scores):
        raw_scores.append(float(score))
        normalized_scores.append(float(score))
        skip_mask.append(bool(agent_skip[idx]))
        unit_meta.append({"type": "agent", "local_index": idx, "masked_unit_size": 1})

    for idx, score in enumerate(map_scores):
        size = int(max(corridor_meta[idx].get("num_polylines", 1), 1))
        raw_scores.append(float(score))
        normalized_scores.append(float(score) / math.sqrt(size))
        skip_mask.append(bool(map_skip[idx]))
        unit_meta.append({"type": "map", "local_index": idx, "masked_unit_size": size})

    return {
        "raw_scores": np.asarray(raw_scores, dtype=np.float32),
        "normalized_scores": np.asarray(normalized_scores, dtype=np.float32),
        "skip_mask": np.asarray(skip_mask, dtype=bool),
        "unit_meta": unit_meta,
    }
