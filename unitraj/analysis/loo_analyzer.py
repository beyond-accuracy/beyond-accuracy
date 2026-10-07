import torch
import numpy as np

from unitraj.analysis.map_grouping import build_lane_corridors
from unitraj.analysis.joint_view import build_joint_view


METRIC_KEYS = ('minADE6', 'minFDE6', 'brier_minFDE6')


def compute_minADE6(traj: torch.Tensor, gt: torch.Tensor) -> float:
    """GT-based minADE over K modes.

    Args:
        traj: [K, T, 2] predicted trajectories (x, y only)
        gt:   [T, 2] ground truth trajectory
    Returns:
        min over K modes of mean L2 distance over T timesteps
    """
    assert traj.dim() == 3, f"traj must be [K, T, 2], got shape {tuple(traj.shape)}"
    ade_per_mode = (traj - gt.to(traj.device).unsqueeze(0)).norm(dim=-1).mean(dim=-1)  # [K]
    return float(ade_per_mode.min().item())


def compute_all_metrics(
    traj: torch.Tensor,
    prob: torch.Tensor,
    gt:   torch.Tensor,
) -> dict:
    """Compute minADE6 / minFDE6 / brier_minFDE6 for one focal agent.

    Args:
        traj: [K, T, 2] predicted trajectories (x, y only)
        prob: [K] predicted probability per mode
        gt:   [T, 2] ground truth trajectory

    minADE6, minFDE6: oracle K=6 trajectory/endpoint errors.
    brier_minFDE6 = minFDE6 + (1 - p(argmin-FDE mode))²
        UniTraj base_model convention (base_model.py:263-265). AV2 leaderboard primary metric.
        Captures both geometric endpoint error AND probability-mass calibration in one signal —
        ΔBrier is sensitive to rank-inversion (p* shift) even when Δgeometry alone is small.
    """
    assert traj.dim() == 3, f"traj must be [K, T, 2], got shape {tuple(traj.shape)}"
    assert prob.dim() == 1, f"prob must be [K], got shape {tuple(prob.shape)}"
    gt_dev = gt.to(traj.device)
    per_mode_ade = (traj - gt_dev.unsqueeze(0)).norm(dim=-1).mean(dim=-1)  # [K]
    per_mode_fde = (traj[:, -1] - gt_dev[-1]).norm(dim=-1)                  # [K]
    best_fde_idx = int(per_mode_fde.argmin().item())
    min_fde6 = float(per_mode_fde[best_fde_idx].item())
    p_at_best_fde = float(prob[best_fde_idx].item())
    return {
        'minADE6':       float(per_mode_ade.min().item()),
        'minFDE6':       min_fde6,
        'brier_minFDE6': min_fde6 + (1.0 - p_at_best_fde) ** 2,
    }


class LOOAnalyzer:
    """Batched Leave-One-Out causal impact analyzer for UniTraj models.

    Batch layout: {'batch_size': int, 'input_dict': {obj_trajs, ...}, 'batch_sample_count': int}
    """

    def __init__(
        self,
        model,
        model_name: str,
        n_agents: int = 32,
        device: str = 'cuda',
        max_corridors: int = 10,
        skip_map: bool = False,
        center_thresh: float = 12.0,
        heading_cos_thresh: float = 0.7,
        endpoint_thresh: float = 6.0,
    ):
        self.model = model
        self.model_name = model_name
        self.n_agents = n_agents
        self.device = device
        self.max_corridors = max_corridors
        self.skip_map = skip_map
        self.center_thresh = center_thresh
        self.heading_cos_thresh = heading_cos_thresh
        self.endpoint_thresh = endpoint_thresh

    def _call_model(self, batch: dict) -> dict:
        """Call the model and return {'predicted_trajectory', 'predicted_probability'}.

        - autobot/wayformer/MTR: forward() → (output_dict, loss_dict) tuple
        - DeMo: forward() → {'y_hat', 'pi', ...} raw dict
        """
        output = self.model(batch)
        if isinstance(output, tuple):
            output = output[0]
        if 'predicted_trajectory' not in output:
            y_hat = output['new_y_hat'] if output.get('new_y_hat') is not None else output['y_hat']
            pi    = output['new_pi']    if output.get('new_pi')    is not None else output['pi']
            output = {
                'predicted_trajectory':  y_hat[..., :2],
                'predicted_probability': torch.softmax(pi.double(), dim=-1),
            }
        return output

    def get_skip_mask(self, batch: dict) -> np.ndarray:
        """Return a boolean mask of agents to skip in LOO.

        Skip conditions:
          1. focal agent (track_index_to_predict)
          2. padding slot (obj_trajs_mask all False)
        """
        inp = batch['input_dict']
        skip = np.zeros(self.n_agents, dtype=bool)

        # 1) focal agents
        focal_indices = inp['track_index_to_predict'][0].cpu().numpy().flatten()
        for idx in focal_indices:
            if 0 <= int(idx) < self.n_agents:
                skip[int(idx)] = True

        obj_mask = inp['obj_trajs_mask'][0]  # [N, T]
        for i in range(min(self.n_agents, obj_mask.shape[0])):
            if not obj_mask[i].any():
                skip[i] = True

        return skip

    def build_loo_batches(self, batch: dict) -> tuple:
        """Return a list of the original + N masked batches.

        Returns:
            tuple: (list of dicts, skip ndarray)
        """
        skip = self.get_skip_mask(batch)
        batches = [batch]  # index 0 = original

        inp = batch['input_dict']
        obj_trajs_orig = inp['obj_trajs']
        obj_mask_orig  = inp['obj_trajs_mask']

        for i in range(self.n_agents):
            if skip[i]:
                batches.append(batch)
            else:
                new_trajs = obj_trajs_orig.clone()
                new_mask  = obj_mask_orig.clone()
                new_trajs[:, i, :, :] = 0.0
                new_mask[:, i, :]     = False
                new_inp   = {**inp, 'obj_trajs': new_trajs, 'obj_trajs_mask': new_mask}
                batches.append({**batch, 'input_dict': new_inp})

        return batches, skip

    def _mask_agents_batch(self, batch: dict, agent_indices: list) -> dict:
        """Return a batch with several agents masked at once."""
        inp = batch['input_dict']
        new_trajs = inp['obj_trajs'].clone()
        new_mask  = inp['obj_trajs_mask'].clone()
        for i in agent_indices:
            new_trajs[:, i, :, :] = 0.0
            new_mask[:, i, :]     = False
        new_inp = {**inp, 'obj_trajs': new_trajs, 'obj_trajs_mask': new_mask}
        return {**batch, 'input_dict': new_inp}

    def _mask_map_corridor_batch(self, batch: dict, polyline_indices: list) -> dict:
        """Return a batch with all polyline slots of a corridor masked."""
        inp = batch['input_dict']
        new_poly = inp['map_polylines'].clone()
        new_pmask = inp['map_polylines_mask'].clone()
        for idx in polyline_indices:
            new_poly[:, idx, :, :]  = 0.0
            new_pmask[:, idx, :]    = False
        new_inp = {**inp, 'map_polylines': new_poly, 'map_polylines_mask': new_pmask}
        return {**batch, 'input_dict': new_inp}

    @torch.no_grad()
    def _compute_faithfulness(
        self,
        causal_vec:       np.ndarray,
        skip:             np.ndarray,
        batch:            dict,
        gt_traj:          torch.Tensor,
        focal_local_idx:  int,
        baseline_metrics: dict,
        scene_idx:        int,
        k:                int = 5,
    ) -> dict:
        """Remove top-k / random-k / bottom-k jointly and compute the 3-metric Δ.

        Ranking uses causal_vec (minADE6 delta), as in the original faithfulness protocol.
        Reports all three of minADE6 / minFDE6 / brier_minFDE6.

        Note: uses _forward_sequential (3 extra forward passes).
        Intentionally differs from the stacked path in analyze_scene — with only 3 passes,
        sequential is simpler and safe for all models (including MTR).
        """
        valid_idx = np.where(~skip)[0]

        sorted_desc = valid_idx[np.argsort(causal_vec[valid_idx])[::-1]]
        top_k_indices = sorted_desc[:min(k, len(sorted_desc))].tolist()

        rng = np.random.default_rng(42 + scene_idx)
        k_rand = min(k, len(valid_idx))
        random_k_indices = rng.choice(valid_idx, size=k_rand, replace=False).tolist()

        neutral_idx = valid_idx[causal_vec[valid_idx] <= 0]
        if len(neutral_idx) > 0:
            sorted_asc = neutral_idx[np.argsort(causal_vec[neutral_idx])]
            bottom_k_indices = sorted_asc[:min(k, len(sorted_asc))].tolist()
        else:
            bottom_k_indices = []

        def _empty():
            return {
                'delta_ade':     None,  # legacy = delta_minADE6
                'delta_fde6':    None,
                'delta_brier':   None,
                'agent_indices': [],
            }

        def _run(indices):
            if not indices:
                return _empty()
            masked = self._mask_agents_batch(batch, indices)
            preds = self._forward_sequential([masked])
            traj = preds[0]['predicted_trajectory'][focal_local_idx][..., :2]
            prob = preds[0]['predicted_probability'][focal_local_idx]
            m = compute_all_metrics(traj, prob, gt_traj)
            return {
                'delta_ade':     float(m['minADE6']       - baseline_metrics['minADE6']),  # legacy
                'delta_fde6':    float(m['minFDE6']       - baseline_metrics['minFDE6']),
                'delta_brier':   float(m['brier_minFDE6'] - baseline_metrics['brier_minFDE6']),
                'agent_indices': list(indices),
            }

        return {
            'top_k':    _run(top_k_indices),
            'random_k': _run(random_k_indices),
            'bottom_k': _run(bottom_k_indices),
        }

    def compute_impact(
        self,
        pred_orig:       dict,
        pred_masked:     dict,
        focal_local_idx: int,
        gt_traj:         torch.Tensor,
    ) -> dict:
        """GT-based causal impact delta for all 5 metrics.

        C_i[metric] = metric(masked, GT) - metric(full, GT)

        C_i > 0: removing the agent worsens the prediction → important agent
        C_i < 0: removing the agent improves the prediction → distractor
        Returns dict with keys in METRIC_KEYS. minADE6 is the legacy signal used for ranking.
        """
        traj_orig = pred_orig['predicted_trajectory'][focal_local_idx][..., :2]   # [K, T, 2]
        traj_mask = pred_masked['predicted_trajectory'][focal_local_idx][..., :2] # [K, T, 2]
        prob_orig = pred_orig['predicted_probability'][focal_local_idx]           # [K]
        prob_mask = pred_masked['predicted_probability'][focal_local_idx]         # [K]
        m_orig = compute_all_metrics(traj_orig, prob_orig, gt_traj)
        m_mask = compute_all_metrics(traj_mask, prob_mask, gt_traj)
        return {k: float(m_mask[k] - m_orig[k]) for k in METRIC_KEYS}

    def compute_causal_vector(
        self,
        predictions_list: list,
        skip:             np.ndarray,
        focal_local_idx:  int,
        gt_traj:          torch.Tensor,
    ) -> dict:
        """Extract the focal agent's GT-based causal impact vectors from N+1 predictions.

        Returns dict {metric_key: np.ndarray[N]} for each metric in METRIC_KEYS.
        Skipped slots are filled with 0.0.
        """
        pred_orig = predictions_list[0]
        vecs = {k: np.zeros(self.n_agents, dtype=np.float32) for k in METRIC_KEYS}

        for i in range(self.n_agents):
            if not skip[i]:
                delta = self.compute_impact(
                    pred_orig, predictions_list[i + 1], focal_local_idx, gt_traj
                )
                for k in METRIC_KEYS:
                    vecs[k][i] = delta[k]
        return vecs

    @torch.no_grad()
    def analyze_scene(self, batch: dict, scene_idx: int = 0) -> dict:
        """Run LOO analysis for every focal agent in one scene."""
        assert self.model is not None
        self.model.eval()

        inp = batch['input_dict']
        scenario_id   = inp.get('scenario_id', ['unknown'])[0]
        focal_indices = inp['track_index_to_predict'][0].cpu().numpy().flatten()
        center_objects_id = inp.get('center_objects_id', None)
        focal_ids = [
            center_objects_id[i] if center_objects_id is not None else str(int(focal_indices[i]))
            for i in range(len(focal_indices))
        ]

        raw_ids = inp.get('obj_trajs_ids', None)
        if raw_ids is not None:
            if isinstance(raw_ids, list):
                raw_ids = raw_ids[0]  # batch_size=1 → first element
            # raw_ids: [max_num_agents] array of str/object
            slot_to_agent_id = [str(raw_ids[i]) for i in range(self.n_agents)]
        else:
            slot_to_agent_id = [str(i) for i in range(self.n_agents)]

        gt_future = inp.get('obj_trajs_future_state', None)

        loo_batches, skip = self.build_loo_batches(batch)

        stacked = self._stack_and_to_device(loo_batches)

        output = self._call_model(stacked)
        pred_traj = output['predicted_trajectory']
        pred_prob = output['predicted_probability']

        n_loo   = len(loo_batches)
        n_focal = len(focal_ids)
        K = pred_traj.shape[1]
        T = pred_traj.shape[2]
        D = pred_traj.shape[3]
        pred_traj = pred_traj.view(n_loo, n_focal, K, T, D)
        pred_prob = pred_prob.view(n_loo, n_focal, K)

        predictions_list = [
            {
                'predicted_trajectory': pred_traj[i],
                'predicted_probability': pred_prob[i],
            }
            for i in range(n_loo)
        ]

        # 5) Map corridor grouping + sequential LOO
        corridors = []
        map_preds_list = []
        map_polylines = inp.get('map_polylines', None)
        if not self.skip_map and map_polylines is not None:
            map_mask_t   = inp.get('map_polylines_mask', None)
            map_center_t = inp.get('map_polylines_center', None)
            if map_mask_t is not None and map_center_t is not None:
                mp_np = map_polylines.cpu().numpy() if isinstance(map_polylines, torch.Tensor) else map_polylines
                mm_np = map_mask_t.cpu().numpy()    if isinstance(map_mask_t,   torch.Tensor) else map_mask_t
                mc_np = map_center_t.cpu().numpy()  if isinstance(map_center_t, torch.Tensor) else map_center_t
                corridors = build_lane_corridors(
                    mp_np, mm_np, mc_np,
                    max_corridors=self.max_corridors,
                    center_thresh=self.center_thresh,
                    heading_cos_thresh=self.heading_cos_thresh,
                    endpoint_thresh=self.endpoint_thresh,
                )
                corridor_batches = [
                    self._mask_map_corridor_batch(batch, c['polyline_indices'])
                    for c in corridors
                ]
                if corridor_batches:
                    map_preds_list = self._forward_sequential(corridor_batches)

        result = {'scenario_id': scenario_id, 'focal_agents': {}}
        for j, agent_id in enumerate(focal_ids):
            focal_slot = int(focal_indices[j])

            # GT: [F, 2], CPU tensor
            if gt_future is not None:
                gt_traj = gt_future[0, focal_slot, :, :2].float().cpu()
            else:
                gt_traj = None

            if gt_traj is not None:
                causal_vecs = self.compute_causal_vector(
                    predictions_list, skip, focal_local_idx=j, gt_traj=gt_traj
                )
                baseline_metrics = compute_all_metrics(
                    predictions_list[0]['predicted_trajectory'][j][..., :2],
                    predictions_list[0]['predicted_probability'][j],
                    gt_traj,
                )
                faith = self._compute_faithfulness(
                    causal_vecs['minADE6'], skip, batch, gt_traj,
                    focal_local_idx=j,
                    baseline_metrics=baseline_metrics,
                    scene_idx=scene_idx,
                )
                # map-corridor causal impact per metric
                map_scores_dict = {k: np.zeros(len(map_preds_list), dtype=np.float32) for k in METRIC_KEYS}
                for mi, pred_masked in enumerate(map_preds_list):
                    delta = self.compute_impact(predictions_list[0], pred_masked, focal_local_idx=j, gt_traj=gt_traj)
                    for k in METRIC_KEYS:
                        map_scores_dict[k][mi] = delta[k]
            else:
                causal_vecs = {k: np.zeros(self.n_agents, dtype=np.float32) for k in METRIC_KEYS}
                baseline_metrics = {k: None for k in METRIC_KEYS}
                faith = None
                map_scores_dict = {k: np.zeros(len(corridors), dtype=np.float32) for k in METRIC_KEYS}

            # Legacy aliases (downstream scripts expect these)
            causal_vec = causal_vecs['minADE6']
            map_scores = map_scores_dict['minADE6']

            map_skip = np.zeros(len(corridors), dtype=bool)

            agent_view = {
                'scores':      causal_vec,          # legacy = minADE6 delta
                'scores_dict': causal_vecs,         # new: dict of 3 metrics (METRIC_KEYS)
                'skip_mask':   skip.copy(),
                'agent_ids':   slot_to_agent_id,
            }
            map_view = {
                'scores':        map_scores,        # legacy = minADE6 delta
                'scores_dict':   map_scores_dict,   # new: dict of 3 metrics (METRIC_KEYS)
                'skip_mask':     map_skip,
                'corridor_meta': corridors,
            }
            joint_view = build_joint_view(causal_vec, skip, map_scores, map_skip, corridors)

            result['focal_agents'][str(agent_id)] = {
                'causal_vector':           causal_vec,           # legacy alias (minADE6 delta)
                'causal_vectors':          causal_vecs,          # new: dict of 3 metric vectors
                'focal_idx':               focal_slot,
                'skip_mask':               skip.copy(),          # legacy alias
                'baseline_minADE6':        baseline_metrics['minADE6'],
                'baseline_minFDE6':        baseline_metrics['minFDE6'],
                'baseline_brier_minFDE6':  baseline_metrics['brier_minFDE6'],
                'faithfulness':            faith,
                'agent_view':              agent_view,
                'map_view':                map_view,
                'joint_view':              joint_view,
            }

        return result

    def _forward_sequential(self, batches: list) -> list:
        """Forward each LOO batch separately — for models without stacked-batch support (e.g. MTR)."""
        predictions = []
        for b in batches:
            single_inp = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in b['input_dict'].items()
            }
            single = {'batch_size': 1, 'input_dict': single_inp, 'batch_sample_count': 1}
            output = self._call_model(single)
            predictions.append({
                'predicted_trajectory': output['predicted_trajectory'],
                'predicted_probability': output['predicted_probability'],
            })
        return predictions

    def _stack_and_to_device(self, batches: list) -> dict:
        """Stack a list of batches into one tensor batch and move it to GPU.

        List fields (center_objects_type, center_objects_id, etc.) must also be repeated n times
        so that num_center_objects inside the model matches the batch count.
        """
        ref_inp = batches[0]['input_dict']
        stacked_inp = {}
        for key, val in ref_inp.items():
            if isinstance(val, torch.Tensor):
                stacked_inp[key] = torch.cat([b['input_dict'][key] for b in batches], dim=0).to(self.device)
            elif isinstance(val, np.ndarray):
                stacked_inp[key] = np.concatenate([b['input_dict'][key] for b in batches], axis=0)
            elif isinstance(val, list):
                combined = []
                for b in batches:
                    combined.extend(b['input_dict'].get(key, []))
                stacked_inp[key] = combined
            else:
                stacked_inp[key] = val

        n = len(batches)
        return {
            'batch_size': n,
            'input_dict': stacked_inp,
            'batch_sample_count': n,
        }
