#!/usr/bin/env python3
"""Build the preprocessed H5 cache without training.

All five models read one shared cache per dataset (the cache path does not depend on the model),
so build it once with method=DeMo, as in the paper.

Usage:
    python tools/build_cache.py method=DeMo dataset=waymo only_train_on_ego=True
    python tools/build_cache.py method=DeMo dataset=av2
    python tools/build_cache.py method=DeMo dataset=av2 +val_only=True   # validation split only (enough for LOO)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'unitraj'))

import hydra
from omegaconf import OmegaConf
from datasets import build_dataset


@hydra.main(version_base=None, config_path='../unitraj/configs', config_name='config')
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg = OmegaConf.merge(cfg, cfg.dataset)

    cfg.use_cache = False
    cfg.overwrite_cache = True

    print(f'Building cache for: {cfg.dataset_name}  |  only_train_on_ego={cfg.get("only_train_on_ego", False)}')
    print(f'Cache path: {cfg.cache_path}')

    if not cfg.get('val_only', False):
        print('\n[1/2] Building train cache...')
        build_dataset(cfg, val=False)
    print('[2/2] Building val cache...')
    build_dataset(cfg, val=True)
    print('\nDone.')


if __name__ == '__main__':
    main()
