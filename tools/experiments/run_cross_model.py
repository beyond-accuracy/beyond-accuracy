#!/usr/bin/env python3
"""Cross-Model Agreement analysis CLI.

Usage:
    python tools/experiments/run_cross_model.py \\
        --loo-dir results/loo/av2 \\
        --output  results/benchmark/av2/cross_model.pkl
"""
import argparse
import pickle
from pathlib import Path

from unitraj.analysis.cross_model_agreement import compute_cross_model_agreement


def main():
    parser = argparse.ArgumentParser(description='Cross-Model Agreement analysis')
    parser.add_argument('--loo-dir', type=Path, required=True,
                        help='Directory containing {model}.pkl files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output PKL path')
    parser.add_argument('--k', type=int, default=5,
                        help='Top-k for Jaccard (default: 5)')
    parser.add_argument('--view', type=str, default=None,
                        choices=['agent_view', 'map_view', 'joint_view'],
                        help='Which view to compare (default: legacy causal_vector)')
    parser.add_argument('--score-key', type=str, default='scores',
                        help='Score key inside the view dict (default: scores)')
    args = parser.parse_args()

    # Load all LOO PKLs from directory
    loo_pkls = {}
    for pkl_path in sorted(args.loo_dir.glob('*.pkl')):
        model_name = pkl_path.stem
        with open(pkl_path, 'rb') as f:
            loo_pkls[model_name] = pickle.load(f)
        print(f'Loaded {model_name}: {len(loo_pkls[model_name]["results"])} scenes')

    if len(loo_pkls) < 2:
        raise ValueError(f'Need at least 2 model PKLs, got {len(loo_pkls)}')

    print(f'\nComputing cross-model agreement (k={args.k}) for models: {list(loo_pkls.keys())}')
    result = compute_cross_model_agreement(loo_pkls, k=args.k, view=args.view, score_key=args.score_key)

    # Print summary
    print('\nJaccard@k (mean over scenes):')
    for pair, score in result['jaccard'].items():
        print(f'  {pair}: {score:.4f}')
    print('\nSpearman r (mean over scenes):')
    for pair, score in result['spearman'].items():
        print(f'  {pair}: {score:.4f}')

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(result, f)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
