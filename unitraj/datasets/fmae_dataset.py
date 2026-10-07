import numpy as np
import torch

from .base_dataset import BaseDataset


def convert_to_fmae_format(input_dict, config):
    """Convert base collate output to ForecastMAE format.

    Standalone function (no model dependency) so it can run in the
    DataLoader prefetch pipeline, overlapping with GPU computation.
    Does NOT call .cuda() — Lightning's device transfer handles that.
    """
    B         = len(input_dict['center_objects_id'])
    N_agent   = input_dict["obj_trajs_mask"].shape[1]
    N_lane    = input_dict["map_polylines"].shape[1]
    h_steps   = config["past_len"]
    f_steps   = config["future_len"]
    lane_pts  = config["max_points_per_lane"]

    # ── Agent position differences ───────────────────────────────────────────
    x = torch.zeros(B, N_agent, h_steps, 2)
    x[..., 1:, :] = torch.where(
        (~input_dict["obj_trajs_mask"][..., :h_steps - 1] |
         ~input_dict["obj_trajs_mask"][..., 1:h_steps]).unsqueeze(-1),
        torch.zeros(B, N_agent, h_steps - 1, 2),
        input_dict["obj_trajs_pos"][..., 1:h_steps, :2] -
        input_dict["obj_trajs_pos"][..., :h_steps - 1, :2],
    )

    # ── Agent attributes (vectorized, replaces B×N_agent Python loop) ────────
    last  = input_dict["obj_trajs"][:, :, h_steps - 1, :]  # (B, N, D)
    first = input_dict["obj_trajs"][:, :, 0, :]             # (B, N, D)

    is_veh  = last[..., 6] == 1
    is_ped  = last[..., 7] == 1
    is_bike = last[..., 8] == 1

    attr0 = torch.full((B, N_agent), 9, dtype=torch.uint8)
    attr0 = torch.where(is_veh,  torch.zeros_like(attr0),        attr0)
    attr0 = torch.where(is_ped,  torch.ones_like(attr0),         attr0)
    attr0 = torch.where(is_bike, torch.full_like(attr0, 3),      attr0)

    attr2 = torch.full((B, N_agent), 3, dtype=torch.uint8)
    attr2 = torch.where(is_veh,  torch.zeros_like(attr2),        attr2)
    attr2 = torch.where(is_ped,  torch.ones_like(attr2),         attr2)
    attr2 = torch.where(is_bike, torch.full_like(attr2, 2),      attr2)

    is_focal  = last[..., 9]  == 1
    is_sdc    = last[..., 10] == 1
    is_scored = ~is_focal & (first[..., 10] == 0)
    attr1 = torch.zeros(B, N_agent, dtype=torch.uint8)
    attr1 = torch.where(is_scored,          torch.full_like(attr1, 2), attr1)
    attr1 = torch.where(is_focal | is_sdc,  torch.full_like(attr1, 3), attr1)

    x_attr = torch.stack([attr0, attr1, attr2], dim=-1)

    # ── Velocity ─────────────────────────────────────────────────────────────
    vel_hist = torch.linalg.norm(
        input_dict["obj_trajs"][..., h_steps + 14:h_steps + 16].float(), dim=-1)
    vel_fut  = torch.linalg.norm(
        input_dict["obj_trajs_future_state"][..., 2:].float(), dim=-1)
    x_velocity = torch.cat([vel_hist, vel_fut], dim=-1)

    x_velocity_diff = torch.zeros(B, N_agent, h_steps)
    x_velocity_diff[..., 1:] = torch.where(
        (~input_dict["obj_trajs_mask"][..., :h_steps - 1] |
         ~input_dict["obj_trajs_mask"][..., 1:h_steps]),
        torch.zeros(B, N_agent, h_steps - 1),
        x_velocity[..., 1:h_steps] - x_velocity[..., :h_steps - 1],
    )

    # ── Lane ─────────────────────────────────────────────────────────────────
    lane_positions = input_dict["map_polylines"][..., :2].clone()
    mid = lane_pts // 2
    lane_centers = lane_positions[:, :, max(mid - 1, 0):mid + 1].mean(dim=-2)
    lane_angles  = torch.atan2(
        lane_positions[..., mid, 1] - lane_positions[..., mid - 1, 1],
        lane_positions[..., mid, 0] - lane_positions[..., mid - 1, 0],
    )
    lane_padding_mask     = ~input_dict["map_polylines_mask"].clone()
    lane_key_padding_mask = lane_padding_mask.all(-1)

    # ── Padding masks ────────────────────────────────────────────────────────
    x_positions    = input_dict["obj_trajs"][..., :h_steps, :2].clone()
    x_padding_mask = torch.cat(
        (~input_dict["obj_trajs_mask"], ~input_dict["obj_trajs_future_mask"]), dim=-1)

    too_far = torch.linalg.norm(
        x_positions[..., h_steps - 1, :].float(), dim=-1) > 150
    x_padding_mask = torch.where(
        too_far.unsqueeze(-1), torch.ones_like(x_padding_mask), x_padding_mask)
    x_padding_mask = torch.where(
        x_padding_mask[..., h_steps - 1].unsqueeze(-1),
        torch.ones_like(x_padding_mask), x_padding_mask)
    x_key_padding_mask = x_padding_mask.all(-1)

    # ── Target trajectories ──────────────────────────────────────────────────
    y = torch.where(
        (~input_dict["obj_trajs_mask"][..., h_steps - 1].unsqueeze(-1) |
         ~input_dict["obj_trajs_future_mask"]).unsqueeze(-1),
        torch.zeros(B, N_agent, f_steps, 2),
        input_dict["obj_trajs_future_state"][..., :f_steps, :2] -
        input_dict["obj_trajs"][..., h_steps - 1, :2].unsqueeze(-2),
    )

    return {
        "x":                    x,
        "x_attr":               x_attr,
        "x_positions":          x_positions,
        "x_centers":            input_dict["obj_trajs"][..., h_steps - 1, :2].clone(),
        "x_angles":             torch.arcsin(input_dict["obj_trajs"][..., h_steps + 12].float()),
        "x_velocity":           x_velocity,
        "x_velocity_diff":      x_velocity_diff,
        "lane_positions":       lane_positions,
        "lane_centers":         lane_centers,
        "lane_angles":          lane_angles,
        "lane_attr":            torch.zeros(B, N_lane, 3),
        "is_intersections":     torch.zeros(B, N_lane),
        "y":                    y,
        "x_padding_mask":       x_padding_mask,
        "lane_padding_mask":    lane_padding_mask,
        "x_key_padding_mask":   x_key_padding_mask,
        "lane_key_padding_mask": lane_key_padding_mask,
        "num_actors":           (~x_key_padding_mask).sum(-1),
        "num_lanes":            (~lane_key_padding_mask).sum(-1),
        "scenario_id":          input_dict["scenario_id"],
        "track_id":             [],
        "origin":               input_dict["center_objects_world"][..., :2].clone(),
        "theta":                input_dict["center_objects_world"][..., 6].clone(),
    }


def convert_to_demo_format(input_dict, config, device=None):
    """Convert base collate output to DeMo format.

    Standalone function (no model dependency) so it can run in the
    DataLoader prefetch pipeline, overlapping with GPU computation.
    Does NOT call .cuda() — Lightning's device transfer handles that.

    Args:
        device: if provided, new tensors are created on this device directly
                (avoids CPU→GPU roundtrip when input_dict is already on GPU).
    """
    B         = len(input_dict['center_objects_id'])
    N_agent   = input_dict["obj_trajs_mask"].shape[1]
    N_lane    = input_dict["map_polylines"].shape[1]
    h_steps   = config["past_len"]
    f_steps   = config["future_len"]
    lane_pts  = config["max_points_per_lane"]

    kw = dict(device=device) if device is not None else {}

    # ── Agent position differences ─────────────────────────────────────────
    x_positions_diff = torch.zeros(B, N_agent, h_steps, 2, **kw)
    x_positions_diff[..., 1:, :] = torch.where(
        (~input_dict["obj_trajs_mask"][..., :h_steps - 1] |
         ~input_dict["obj_trajs_mask"][..., 1:h_steps]).unsqueeze(-1),
        torch.zeros(B, N_agent, h_steps - 1, 2, **kw),
        input_dict["obj_trajs_pos"][..., 1:h_steps, :2] -
        input_dict["obj_trajs_pos"][..., :h_steps - 1, :2],
    )

    # ── Masks (valid = positive) ───────────────────────────────────────────
    x_valid_mask = input_dict["obj_trajs_mask"][..., :h_steps].clone()
    x_centers = input_dict["obj_trajs_pos"][..., h_steps - 1, :2].clone()
    too_far = torch.linalg.norm(x_centers.float(), dim=-1) > 150
    no_hist = ~input_dict["obj_trajs_mask"][..., h_steps - 1]
    x_key_valid_mask = ~(too_far | no_hist)

    # ── Velocity ───────────────────────────────────────────────────────────
    vel_hist = torch.linalg.norm(
        input_dict["obj_trajs"][..., h_steps + 14:h_steps + 16].float(), dim=-1)
    vel_fut = torch.linalg.norm(
        input_dict["obj_trajs_future_state"][..., 2:].float(), dim=-1)
    x_velocity = torch.cat([vel_hist, vel_fut], dim=-1)

    x_velocity_diff = torch.zeros(B, N_agent, h_steps, **kw)
    x_velocity_diff[..., 1:] = torch.where(
        (~input_dict["obj_trajs_mask"][..., :h_steps - 1] |
         ~input_dict["obj_trajs_mask"][..., 1:h_steps]),
        torch.zeros(B, N_agent, h_steps - 1, **kw),
        x_velocity[..., 1:h_steps] - x_velocity[..., :h_steps - 1],
    )

    # ── Angles ─────────────────────────────────────────────────────────────
    x_angles = torch.arcsin(input_dict["obj_trajs"][..., h_steps + 12].float())

    # ── Agent attributes (vectorized) ──────────────────────────────────────
    last = input_dict["obj_trajs"][:, :, h_steps - 1, :]  # (B, N, D)

    is_veh  = last[..., 6] == 1
    is_ped  = last[..., 7] == 1
    is_bike = last[..., 8] == 1

    attr0 = torch.full((B, N_agent), 9, dtype=torch.long, **kw)
    attr0 = torch.where(is_veh,  torch.zeros_like(attr0),   attr0)
    attr0 = torch.where(is_ped,  torch.ones_like(attr0),    attr0)
    attr0 = torch.where(is_bike, torch.full_like(attr0, 3), attr0)

    attr2 = torch.full((B, N_agent), 3, dtype=torch.long, **kw)
    attr2 = torch.where(is_veh,  torch.zeros_like(attr2),   attr2)
    attr2 = torch.where(is_ped,  torch.ones_like(attr2),    attr2)
    attr2 = torch.where(is_bike, torch.full_like(attr2, 2), attr2)

    # attr1 is always 0 in original DeMo convert (no focal/sdc logic)
    attr1 = torch.zeros(B, N_agent, dtype=torch.long, **kw)

    x_attr = torch.stack([attr0, attr1, attr2], dim=-1)

    # ── Lane features ──────────────────────────────────────────────────────
    lane_positions  = input_dict["map_polylines"][..., :2].clone()
    lane_valid_mask = input_dict["map_polylines_mask"].clone()
    lane_key_valid_mask = ~(~lane_valid_mask).all(-1)

    mid = lane_pts // 2
    lane_centers = lane_positions[:, :, max(mid - 1, 0):mid + 1].mean(dim=-2)
    lane_angles = torch.atan2(
        lane_positions[:, :, mid, 1] - lane_positions[:, :, max(mid - 1, 0), 1],
        lane_positions[:, :, mid, 0] - lane_positions[:, :, max(mid - 1, 0), 0],
    )
    lane_attr = torch.zeros(B, N_lane, 3, dtype=torch.long, **kw)

    # ── Target trajectories ────────────────────────────────────────────────
    last_pos = input_dict["obj_trajs_pos"][..., h_steps - 1, :2]
    target = torch.where(
        (~input_dict["obj_trajs_mask"][..., h_steps - 1].unsqueeze(-1) |
         ~input_dict["obj_trajs_future_mask"]).unsqueeze(-1),
        torch.zeros(B, N_agent, f_steps, 2, **kw),
        input_dict["obj_trajs_future_state"][..., :f_steps, :2] -
        last_pos.unsqueeze(-2),
    )
    target_mask = input_dict["obj_trajs_future_mask"].clone()

    return {
        "x_valid_mask":       x_valid_mask,
        "x_key_valid_mask":   x_key_valid_mask,
        "x_positions_diff":   x_positions_diff,
        "x_velocity_diff":    x_velocity_diff,
        "x_centers":          x_centers,
        "x_angles":           x_angles,
        "x_attr":             x_attr,
        "lane_positions":     lane_positions,
        "lane_valid_mask":    lane_valid_mask.float(),
        "lane_key_valid_mask": lane_key_valid_mask,
        "lane_centers":       lane_centers,
        "lane_angles":        lane_angles,
        "lane_attr":          lane_attr,
        "target":             target,
        "target_mask":        target_mask,
    }


class FMAEDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        model_name = config.get("model_name", "") if config else ""
        self._convert_demo = model_name == "DeMo"

    def collate_fn(self, data_list):
        batch = super().collate_fn(data_list)
        if self._convert_demo:
            batch['demo_data'] = convert_to_demo_format(batch['input_dict'], self.config)
        else:
            batch['fmae_data'] = convert_to_fmae_format(batch['input_dict'], self.config)
        return batch
