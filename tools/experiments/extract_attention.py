#!/usr/bin/env python3
"""Attention weight extraction script.

Runs one original forward pass per scene, extracting
agent-to-agent attention weights and saving them to a pkl.

Used to compare LOO scores (behavioral) with attention weights (internal).

Usage:
    python tools/experiments/extract_attention.py \
        --model DeMo \
        --dataset waymo \
        --ckpt /path/to/checkpoint.ckpt \
        --output results/attention/waymo/DeMo.pkl \
        [--max-scenes 100]

    # All 5 models at once (use run_all_attention.sh)
"""
import argparse
import pickle
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / 'unitraj'))



class AttentionHook:
    """Generic class that captures attention weights via forward hooks.

    Monkey-patches the forward of torch.nn.MultiheadAttention to
    call it with need_weights=True and stores the result.
    """

    def __init__(self):
        self.captured = {}  # layer_name → attention weight tensor
        self._hooks = []
        self._originals = {}

    def _make_hook(self, name):
        """Create a hook function that captures attention weights."""
        captured = self.captured

        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                captured[name] = output[1].detach().cpu()

        return hook_fn

    def _patch_mha_need_weights(self, module, name):
        """Patch the forward of nn.MultiheadAttention to always use need_weights=True."""
        original_forward = module.forward

        def patched_forward(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = True  # [B, tgt_len, src_len]
            return original_forward(*args, **kwargs)

        self._originals[name] = original_forward
        module.forward = patched_forward

    def _get_inner_model(self, trainer_model):
        """Extract the actual forecasting model from a Pytorch Lightning trainer wrapper."""
        for attr in ('net', 'model', 'forecaster', 'encoder'):
            if hasattr(trainer_model, attr):
                inner = getattr(trainer_model, attr)
                if isinstance(inner, torch.nn.Module):
                    return inner
        return trainer_model

    def register(self, model, model_name: str):
        """Register hooks on the appropriate attention layers for each model.

        DeMo/ForecastMAE/AutoBot: torch.nn.MultiheadAttention → patch + hook
        Wayformer: custom MultiHeadAttention (capital H) → patch with a re-implemented forward
        MTR: uses local attention → skip (recorded as extraction failure)
        """
        self.captured.clear()

        for name, child in model.named_modules():
            if isinstance(child, torch.nn.MultiheadAttention):
                self._patch_mha_need_weights(child, name)
                handle = child.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)

        for name, child in model.named_modules():
            cls = child.__class__.__name__
            if cls == 'MultiHeadAttention' and not isinstance(child, torch.nn.MultiheadAttention):
                self._patch_wayformer_custom(child, f'wayformer:{name}')
                self._hooks.append(
                    child.register_forward_hook(self._make_hook_wayformer(f'wayformer:{name}'))
                )


    def _patch_wayformer_custom(self, module, name):
        """Patch the forward of Wayformer custom MultiHeadAttention to store attn weights.

        Runs the original forward, then recomputes Q/K to build the attention matrix.
        pad_mask is broadcast [B, j] → [B, 1, 1, j].
        """
        from einops import rearrange

        original_forward = module.forward
        captured = self.captured
        hook_self = self

        def patched_forward(x_q, x_kv=None, pad_mask=None, rot_pos_emb_q=None, rot_pos_emb_k=None, kv_cache=None):
            if x_kv is not None:
                output = original_forward(
                    x_q, x_kv, pad_mask=pad_mask,
                    rot_pos_emb_q=rot_pos_emb_q, rot_pos_emb_k=rot_pos_emb_k, kv_cache=kv_cache,
                )
            else:
                output = original_forward(
                    x_q, x_q, pad_mask=pad_mask,
                    rot_pos_emb_q=rot_pos_emb_q, rot_pos_emb_k=rot_pos_emb_k, kv_cache=kv_cache,
                )

            try:
                with torch.no_grad():
                    kv_input = x_q if x_kv is None else x_kv
                    q = module.q_proj(x_q)
                    k = module.k_proj(kv_input)
                    q = rearrange(q, "b n (h c) -> b h n c", h=module.num_heads)
                    k = rearrange(k, "b n (h c) -> b h n c", h=module.num_heads)
                    q = q * module.dp_scale

                    attn = torch.einsum("b h i c, b h j c -> b h i j", q, k)

                    if pad_mask is not None:
                        # pad_mask: [B, j] → [B, 1, 1, j]
                        pm = pad_mask
                        if pm.dim() == 2:
                            pm = pm.unsqueeze(1).unsqueeze(1)
                        attn_max_neg = -torch.finfo(attn.dtype).max
                        attn = attn.masked_fill(pm, attn_max_neg)

                    attn = attn.softmax(dim=-1)  # [B, H, Q, K]
                    attn_mean = attn.mean(dim=1)  # [B, Q, K]
                    captured[name] = attn_mean.detach().cpu()
            except Exception as e:
                if not hasattr(hook_self, '_wayformer_errors'):
                    hook_self._wayformer_errors = 0
                hook_self._wayformer_errors += 1
                if hook_self._wayformer_errors <= 3:
                    print(f'[Wayformer] Failed to capture {name}: {e}')

            return output

        self._originals[name] = original_forward
        module.forward = patched_forward

    def _make_hook_wayformer(self, name):
        """No-op (patched_forward writes to captured directly)."""
        def hook_fn(module, input, output):
            pass
        return hook_fn

    def remove(self):
        """Remove all hooks and restore the original forward."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self.captured.clear()

    def get_agent_attention_batch(
        self, n_agents: int, model_name: str, expected_batch: int | None = None
    ) -> np.ndarray | None:
        """Extract the agent-to-agent part of the captured attention per batch.

        Returns: [B, n_agents, n_agents] or None

        Per-model handling:
          DeMo/forecast: net.blocks.*.attn → [B, N+M, N+M] (agent+lane self-attn)
          autobot: social_attn_layers.*.layers.*.self_attn → [B*T, N+1, N+1]
                   → reshape to [B, T, N+1, N+1] → mean over T
          wayformer: perceiver self/cross attn (latent-to-agent)
                    → currently simplified: no agent-to-agent part; with perceiver_encoder
                    → cross_attn is [B, L, N_tokens] (L=latents, N=agents+roads)
                    → slice agents [:n_agents] to get [B, L, n] → mean over latents
          MTR: uses local attention → None
        """
        if not self.captured:
            return None

        # ── Scene encoder layer filter ──
        def is_scene_encoder_layer(key: str) -> bool:
            k = key.lower()
            if model_name == 'DeMo':
                return 'net.blocks.' in k and '.attn' in k and 'time_decoder' not in k and 'mode' not in k
            if model_name == 'forecast':
                return '.blocks.' in k and '.attn' in k and 'decoder' not in k and 'mode' not in k
            if model_name == 'autobot':
                return 'social_attn_layers' in k and '.self_attn' in k
            if model_name == 'wayformer':
                return 'wayformer:' in key and 'encoder' in k
            if model_name == 'MTR':
                return False  # local attention → skip
            return True

        scene_keys = [k for k in self.captured.keys() if is_scene_encoder_layer(k)]
        if not scene_keys:
            return None

        # ── AutoBot: social attention shape = [B*T, N+1, N+1] ──
        if model_name == 'autobot':
            return self._extract_autobot(scene_keys, n_agents, expected_batch)

        # ── Wayformer: perceiver self/cross attention ──
        if model_name == 'wayformer':
            return self._extract_wayformer(scene_keys, n_agents, expected_batch)

        # ── DeMo / ForecastMAE: self-attention over [B, N+M, N+M] ──
        max_size = max(self._attn_side_size(self.captured[k]) for k in scene_keys)
        scene_keys = [k for k in scene_keys if self._attn_side_size(self.captured[k]) == max_size]

        attns = []
        for k in scene_keys:
            a = self.captured[k]
            if a.dim() == 4:
                a = a.mean(dim=1)
            if a.dim() == 2:
                a = a.unsqueeze(0)
            attns.append(a)
        attn = torch.stack(attns, dim=0).mean(dim=0)  # [B, T, S]
        B, T, S = attn.shape
        n = min(n_agents, T, S)
        return attn[:, :n, :n].numpy().astype(np.float32)

    def _extract_autobot(self, scene_keys, n_agents, expected_batch):
        """AutoBot social attention: [B*T, N+1, N+1] → [B, n_agents, n_agents].

        AutoBot prepends the ego as agent_0, giving N+1 tokens.
        Here we simply take the first n_agents.
        """
        attns = []
        for k in scene_keys:
            a = self.captured[k]  # [B*T, N+1, N+1]
            if a.dim() != 3:
                continue
            BT = a.shape[0]
            if expected_batch is None or BT % expected_batch != 0:
                a = a.mean(dim=0, keepdim=True)  # [1, N+1, N+1]
            else:
                T = BT // expected_batch
                a = a.view(expected_batch, T, a.shape[1], a.shape[2]).mean(dim=1)  # [B, N+1, N+1]
            attns.append(a)

        if not attns:
            return None
        max_size = max(a.shape[-1] for a in attns)
        attns = [a for a in attns if a.shape[-1] == max_size]
        attn = torch.stack(attns, dim=0).mean(dim=0)  # [B, N+1, N+1]
        B, T, S = attn.shape
        n = min(n_agents, T, S)
        return attn[:, :n, :n].numpy().astype(np.float32)

    def _extract_wayformer(self, scene_keys, n_agents, expected_batch):
        """Wayformer perceiver cross-attention: [B, L, num_tokens] → [B, n, n].

        In Wayformer, learned latents cross-attend to agent+road tokens.
        Agent importance is the "total latent attention on each agent":
          attention[:, :, agent_i] averaged over the latent dim (L) → [B, n_agents]

        The agent-to-agent matrix is not directly available, so this vector is
        broadcast to every row to form [B, n, n] (outer product of agent importance).
        """
        _ = expected_batch  # reserved
        attns = []
        for k in scene_keys:
            a = self.captured[k]  # [B, L, T] or [B, T, S]
            if a.dim() != 3:
                continue
            attns.append(a)

        if not attns:
            return None

        # Cross-attention: non-square (L ≠ num_tokens)
        cross_attns = [a for a in attns if a.shape[1] != a.shape[2]]
        if not cross_attns:
            a = attns[0]
            B = a.shape[0]
            return np.eye(n_agents, dtype=np.float32)[None].repeat(B, axis=0)

        max_tokens = max(a.shape[-1] for a in cross_attns)
        cross_attns = [a for a in cross_attns if a.shape[-1] == max_tokens]
        attn = torch.stack(cross_attns, dim=0).mean(dim=0)  # [B, L, num_tokens]

        agent_imp = attn.mean(dim=1)  # [B, num_tokens]
        B = agent_imp.shape[0]
        n = min(n_agents, agent_imp.shape[-1])
        agent_imp = agent_imp[:, :n]  # [B, n]

        out = agent_imp.unsqueeze(1).expand(B, n, n).numpy().astype(np.float32)
        return out

    @staticmethod
    def _attn_side_size(attn_tensor) -> int:
        """Return the attention tensor's sequence length ([*, T, S] → max(T, S))."""
        if attn_tensor.dim() == 4:
            return max(attn_tensor.shape[-2], attn_tensor.shape[-1])
        if attn_tensor.dim() == 3:
            return max(attn_tensor.shape[-2], attn_tensor.shape[-1])
        if attn_tensor.dim() == 2:
            return max(attn_tensor.shape[-2], attn_tensor.shape[-1])
        return 0


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='Attention Weight Extraction')
    parser.add_argument('--model',      required=True, help='DeMo | forecast | wayformer | autobot | MTR')
    parser.add_argument('--dataset',    required=True, help='waymo | argoverse2')
    parser.add_argument('--ckpt',       required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--output',     required=True, help='Output pkl path')
    parser.add_argument('--max-scenes', type=int, default=None)
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--n-agents',   type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=16, help='Eval batch size (inference only, attention hook)')
    parser.add_argument('--num-workers',type=int, default=4)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--sdc-only',   action='store_true')
    return parser.parse_args()


def load_config(model_name: str, dataset_name: str, batch_size: int = 16):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    config_dir = str(Path(__file__).parent.parent.parent / 'unitraj' / 'configs')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name='config',
            overrides=[f'method={model_name}', f'dataset={dataset_name}', 'use_cache=True'],
        )
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg = OmegaConf.merge(cfg, cfg.dataset)
    cfg.max_num_agents = 32
    cfg.train_batch_size = batch_size
    cfg.eval_batch_size = batch_size
    return cfg


def get_scenario_id_batch(batch) -> list[str]:
    """Extract the scenario_id list from a batch ([B] list[str])."""
    input_dict = batch.get('input_dict', batch)
    sids = input_dict.get('scenario_id', None)
    if sids is None:
        return ['unknown']

    out = []
    if isinstance(sids, (list, tuple)):
        for s in sids:
            if isinstance(s, torch.Tensor):
                out.append(str(s.item()))
            elif isinstance(s, np.ndarray):
                out.append(str(s.item() if s.size == 1 else s.tolist()))
            else:
                out.append(str(s))
    elif isinstance(sids, torch.Tensor):
        out = [str(int(x)) for x in sids]
    else:
        out = [str(sids)]
    return out


def get_agent_ids_batch(batch, n_agents: int) -> list[list[str]]:
    """Extract agent ID lists from a batch ([B] list of list[str]).

    - Waymo: input_dict['obj_trajs_ids'] = list of ndarrays (track IDs)
    - AV2: no obj_trajs_ids → use slot index (consistent with the LOO pkl)
    """
    input_dict = batch.get('input_dict', batch)
    _B = batch.get('batch_size', 1)
    if hasattr(_B, 'item'):
        _B = _B.item()
    B = int(_B)

    obj_ids = input_dict.get('obj_trajs_ids', None)
    if obj_ids is None:
        obj_ids = input_dict.get('obj_ids', None)
    if obj_ids is None:
        # Fallback: slot index for each scene
        default_ids = [str(i) for i in range(n_agents)]
        return [default_ids for _ in range(B)]

    out = []
    if isinstance(obj_ids, torch.Tensor):
        for b in range(obj_ids.shape[0]):
            out.append([str(int(x)) for x in obj_ids[b, :n_agents]])
    elif isinstance(obj_ids, (list, tuple)):
        for item in obj_ids:
            if hasattr(item, 'tolist'):
                item = item.tolist()
            out.append([str(x) for x in list(item)[:n_agents]])
    elif isinstance(obj_ids, np.ndarray):
        for b in range(obj_ids.shape[0]):
            row = obj_ids[b] if obj_ids.ndim > 1 else obj_ids
            out.append([str(x) for x in list(row)[:n_agents]])
    return out


def get_valid_mask_batch(batch, n_agents: int) -> np.ndarray:
    """Extract the agent valid mask from a batch ([B, N] bool)."""
    input_dict = batch.get('input_dict', batch)
    mask = input_dict.get('obj_trajs_mask', None)
    if mask is not None and isinstance(mask, torch.Tensor):
        # [B, N, T] → any timestep valid
        return mask[:, :n_agents].any(dim=-1).cpu().numpy()  # [B, N]
    return np.ones((1, n_agents), dtype=bool)


def main():
    args = parse_args()

    print(f'\n=== Attention Extraction: {args.model} on {args.dataset} ===')
    print(f'Checkpoint : {args.ckpt}')
    print(f'Output     : {args.output}')
    if args.max_scenes:
        print(f'Max scenes : {args.max_scenes}')
    print()

    cfg = load_config(args.model, args.dataset, batch_size=args.batch_size)
    cfg.seed = args.seed

    from unitraj.utils.utils import set_seed
    set_seed(args.seed)

    if args.sdc_only:
        cfg.only_train_on_ego = True
        print('SDC-only mode enabled')

    print(f'Batch size : {args.batch_size}')

    from unitraj.datasets import build_dataset
    from unitraj.models import __all__ as MODEL_REGISTRY
    from torch.utils.data import DataLoader

    val_dataset = build_dataset(cfg, val=True)
    loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == 'cuda'),
        collate_fn=val_dataset.collate_fn,
    )

    ModelClass = MODEL_REGISTRY[cfg.method.model_name]
    model = ModelClass.load_from_checkpoint(args.ckpt, config=cfg, map_location=args.device)
    model = model.to(args.device)
    model.eval()

    hook = AttentionHook()
    hook.register(model, args.model)
    print(f'Hooks registered: {len(hook._hooks)} layers')

    # ── Run ──
    all_results = {}
    n_captured = 0
    n_failed = 0
    total_scenes = args.max_scenes or len(val_dataset)

    pbar = tqdm(loader, desc='Extracting attention', total=(total_scenes + args.batch_size - 1) // args.batch_size)
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if args.max_scenes and n_captured >= args.max_scenes:
                break

            try:
                hook.captured.clear()

                # Move to device
                input_dict = batch.get('input_dict', batch)
                batch_gpu = {
                    'batch_size': batch.get('batch_size', args.batch_size),
                    'input_dict': {
                        k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                        for k, v in input_dict.items()
                    },
                    'batch_sample_count': batch.get('batch_sample_count', args.batch_size),
                }

                # Expected batch size from input (for AutoBot T reshape)
                expected_B = batch.get('batch_size', args.batch_size)
                if hasattr(expected_B, 'item'):
                    expected_B = int(expected_B)

                # Forward pass (attention captured via hooks)
                model(batch_gpu)

                # Extract attention: [B, n, n]
                agent_attn_batch = hook.get_agent_attention_batch(
                    args.n_agents, args.model, expected_batch=expected_B
                )
                if agent_attn_batch is None:
                    n_failed += expected_B
                    continue

                B = agent_attn_batch.shape[0]

                # Scene metadata (batched)
                scenario_ids = get_scenario_id_batch(batch_gpu)
                agent_ids_list = get_agent_ids_batch(batch_gpu, args.n_agents)
                valid_mask_batch = get_valid_mask_batch(batch_gpu, args.n_agents)

                # B mismatch: fallback to the smaller
                B = min(B, len(scenario_ids), len(agent_ids_list), valid_mask_batch.shape[0])

                # Dispatch each scene in batch
                for b in range(B):
                    sid = scenario_ids[b] if b < len(scenario_ids) else f'unknown_{batch_idx}_{b}'
                    aids = agent_ids_list[b] if b < len(agent_ids_list) else [str(i) for i in range(args.n_agents)]
                    vmask = valid_mask_batch[b] if b < valid_mask_batch.shape[0] else np.ones(args.n_agents, dtype=bool)

                    all_results[sid] = {
                        'scenario_id': sid,
                        'agent_attention': agent_attn_batch[b],  # [n, n]
                        'agent_ids': aids,
                        'valid_mask': vmask,
                    }
                    n_captured += 1

                    if args.max_scenes and n_captured >= args.max_scenes:
                        break

                pbar.set_postfix(captured=n_captured, failed=n_failed)

            except Exception as e:
                import traceback
                n_failed += args.batch_size
                if n_failed <= 5 * args.batch_size:
                    print(f'\n[WARN] batch {batch_idx} failed: {e}')
                    traceback.print_exc()
                continue

    hook.remove()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': args.model,
        'dataset': args.dataset,
        'n_agents': args.n_agents,
        'results': all_results,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)

    print(f'\nDone. {n_captured} scenes captured, {n_failed} failed → {output_path}')


if __name__ == '__main__':
    main()
