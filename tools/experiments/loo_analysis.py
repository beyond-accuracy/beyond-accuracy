#!/usr/bin/env python3
"""LOO Causal Impact Analysis Script.

Usage:
    python tools/experiments/loo_analysis.py \
        --model autobot \
        --dataset waymo \
        --ckpt /path/to/checkpoint.ckpt \
        --output results/loo/waymo/autobot.pkl \
        [--max-scenes 100]
"""
import argparse
import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / 'unitraj'))  # for legacy 'from utils.xxx' imports

from unitraj.analysis import LOOAnalyzer
from unitraj.utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='LOO Causal Impact Analysis')
    parser.add_argument('--model',      required=True, help='autobot | wayformer | MTR | DeMo | forecast')
    parser.add_argument('--dataset',    required=True, help='waymo | argoverse2')
    parser.add_argument('--ckpt',       required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--output',     required=True, help='Output pkl path')
    parser.add_argument('--max-scenes', type=int, default=None, help='Limit scene count (for debugging)')
    parser.add_argument('--device',     default='cuda', help='cuda | cpu')
    parser.add_argument('--n-agents',      type=int, default=32)
    parser.add_argument('--max-corridors', type=int, default=10, help='Max lane corridors per scene for map LOO')
    parser.add_argument('--seed',          type=int, default=42, help='Random seed for deterministic scene order')
    parser.add_argument('--sdc-only',      action='store_true', help='Fix the SDC (AV) as the focal agent (for Waymo causal GT alignment)')
    return parser.parse_args()


def load_config(model_name: str, dataset_name: str):
    """Load the Hydra config and merge method/dataset into the top level, as in train.py."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    config_dir = str(Path(__file__).parent.parent.parent / 'unitraj' / 'configs')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name='config',
            overrides=[
                f'method={model_name}',
                f'dataset={dataset_name}',
                'use_cache=True',
            ]
        )
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg = OmegaConf.merge(cfg, cfg.dataset)
    cfg.max_num_agents = 32
    cfg.train_batch_size = 1
    cfg.eval_batch_size = 1
    return cfg


def apply_sdc_only(cfg):
    """only_train_on_ego=True → the SDC (AV) is the only focal agent."""
    cfg.only_train_on_ego = True


def main():
    args = parse_args()

    print(f'\n=== LOO Analysis: {args.model} on {args.dataset} ===')
    print(f'Checkpoint : {args.ckpt}')
    print(f'Output     : {args.output}')
    print(f'N agents   : {args.n_agents}')
    print(f'Seed       : {args.seed}')
    if args.max_scenes:
        print(f'Max scenes : {args.max_scenes}')
    print()

    cfg = load_config(args.model, args.dataset)
    cfg.seed = args.seed
    set_seed(args.seed)
    if args.sdc_only:
        apply_sdc_only(cfg)
        print('SDC-only mode: only_train_on_ego=True')

    from unitraj.datasets import build_dataset
    from unitraj.models import build_model
    from torch.utils.data import DataLoader

    val_dataset = build_dataset(cfg, val=True)
    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=(args.device == 'cuda'),
        collate_fn=val_dataset.collate_fn,
    )

    from unitraj.models import __all__ as MODEL_REGISTRY
    ModelClass = MODEL_REGISTRY[cfg.method.model_name]
    model = ModelClass.load_from_checkpoint(args.ckpt, config=cfg, map_location=args.device)
    model = model.to(args.device)
    model.eval()

    analyzer = LOOAnalyzer(
        model=model,
        model_name=args.model,
        n_agents=args.n_agents,
        device=args.device,
        max_corridors=args.max_corridors,
    )

    all_results = {}
    for scene_idx, batch in enumerate(tqdm(loader, desc='LOO Analysis')):
        if args.max_scenes and scene_idx >= args.max_scenes:
            break
        try:
            result = analyzer.analyze_scene(batch, scene_idx=scene_idx)
            all_results[result['scenario_id']] = result
        except Exception as e:
            print(f'\n[WARN] scene {scene_idx} failed: {e}')
            continue

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model':             args.model,
        'dataset':           args.dataset,
        'max_num_agents':    args.n_agents,
        'max_num_corridors': args.max_corridors,
        'results':           all_results,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)

    print(f'\nDone. {len(all_results)} scenes saved → {output_path}')


if __name__ == '__main__':
    main()
