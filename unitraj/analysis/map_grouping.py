"""
map_grouping.py
---------------
Utilities for grouping map polylines into lane corridors.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any

# Lane type IDs (one-hot slot indices in polyline_feat[0, 9:29])
LANE_TYPE_IDS = {1, 2, 3}


def extract_polyline_type_id(polyline_feat: np.ndarray) -> int:
    """Return type_id from polyline_feat[0, 9:29] argmax."""
    return int(np.argmax(polyline_feat[0, 9:29]))


def representative_heading(polyline_xy: np.ndarray) -> np.ndarray:
    """
    Return unit vector from first to last point of polyline_xy (shape [P, 2]).
    Denominator is clamped to at least 1e-6.
    """
    direction = polyline_xy[-1] - polyline_xy[0]
    norm = max(np.linalg.norm(direction), 1e-6)
    return direction / norm


def endpoint_gap(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """
    Return min(|a_end - b_start|, |b_end - a_start|).
    poly_a, poly_b: shape [P, 2] (xy coordinates).
    """
    a_end = poly_a[-1]
    a_start = poly_a[0]
    b_end = poly_b[-1]
    b_start = poly_b[0]
    gap_ab = float(np.linalg.norm(a_end - b_start))
    gap_ba = float(np.linalg.norm(b_end - a_start))
    return min(gap_ab, gap_ba)


def build_lane_corridors(
    map_polylines: np.ndarray,
    map_polylines_mask: np.ndarray,
    map_polylines_center: np.ndarray,
    max_corridors: int = 10,
    center_thresh: float = 12.0,
    heading_cos_thresh: float = 0.7,
    endpoint_thresh: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Group lane polylines into corridors via connected components.

    Parameters
    ----------
    map_polylines : np.ndarray, shape [1, N, P, feat_dim]
    map_polylines_mask : np.ndarray (bool), shape [1, N, P]
    map_polylines_center : np.ndarray, shape [1, N, 3]
    max_corridors : int
        Maximum number of corridors to return.
    center_thresh : float
        Max distance between polyline centers to consider a connection.
    heading_cos_thresh : float
        Min cosine similarity between headings to consider a connection.
    endpoint_thresh : float
        Max endpoint gap to consider a connection.

    Returns
    -------
    List of corridor dicts, sorted by min_distance_to_focal ascending,
    up to max_corridors entries.
    """
    # Use batch index 0
    polys = map_polylines[0]          # [N, P, feat_dim]
    masks = map_polylines_mask[0]     # [N, P]
    centers = map_polylines_center[0] # [N, 3]

    N = polys.shape[0]

    # --- Filter valid lane polylines ---
    valid_lane_indices = []
    for i in range(N):
        # Must have at least one valid point
        if not masks[i].any():
            continue
        type_id = extract_polyline_type_id(polys[i])
        if type_id in LANE_TYPE_IDS:
            valid_lane_indices.append(i)

    if not valid_lane_indices:
        return []

    # Precompute headings and xy arrays for valid lanes
    headings: Dict[int, np.ndarray] = {}
    xy_coords: Dict[int, np.ndarray] = {}
    for i in valid_lane_indices:
        valid_pts = polys[i][masks[i]]  # [K, feat_dim]
        xy = valid_pts[:, :2]
        xy_coords[i] = xy
        headings[i] = representative_heading(xy)

    # --- Build adjacency via pairwise checks ---
    adj: Dict[int, List[int]] = {i: [] for i in valid_lane_indices}
    vl = valid_lane_indices
    for a_idx in range(len(vl)):
        i = vl[a_idx]
        for b_idx in range(a_idx + 1, len(vl)):
            j = vl[b_idx]
            # Center distance check
            ci = centers[i, :2]
            cj = centers[j, :2]
            center_dist = float(np.linalg.norm(ci - cj))
            if center_dist > center_thresh:
                continue
            # Heading cosine check
            cos_sim = float(np.dot(headings[i], headings[j]))
            if cos_sim < heading_cos_thresh:
                continue
            # Endpoint gap check
            gap = endpoint_gap(xy_coords[i], xy_coords[j])
            if gap > endpoint_thresh:
                continue
            adj[i].append(j)
            adj[j].append(i)

    # --- Connected components via DFS ---
    visited = set()
    components: List[List[int]] = []

    for start in valid_lane_indices:
        if start in visited:
            continue
        # DFS
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(sorted(component))

    # --- Build corridor dicts ---
    corridors: List[Dict[str, Any]] = []
    for comp in components:
        comp_centers = centers[comp, :2]  # [K, 2]
        centroid = comp_centers.mean(axis=0)

        # min_distance to focal origin (0, 0)
        dists_to_focal = np.linalg.norm(comp_centers, axis=1)
        min_distance = float(dists_to_focal.min())

        # num_valid_points: sum of valid mask points over all polylines in corridor
        num_valid_points = int(sum(masks[i].sum() for i in comp))

        corridors.append({
            "corridor_id": len(corridors),
            "polyline_indices": comp,
            "centroid": centroid,
            "min_distance_to_focal": min_distance,
            "num_polylines": len(comp),
            "num_valid_points": num_valid_points,
        })

    # --- Sort by min_distance_to_focal ascending ---
    corridors.sort(key=lambda c: c["min_distance_to_focal"])

    # Re-assign corridor_id after sorting
    for cid, corridor in enumerate(corridors):
        corridor["corridor_id"] = cid

    return corridors[:max_corridors]
