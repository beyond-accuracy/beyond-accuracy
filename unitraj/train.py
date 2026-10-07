import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader

import hydra
torch.set_float32_matmul_precision('medium')

from datasets import build_dataset
from models import build_model
from utils.utils import find_latest_checkpoint, get_git_info, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg = OmegaConf.merge(cfg, cfg.dataset)

    run_dir = HydraConfig.get().runtime.output_dir

    # ── File logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(run_dir, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger(__name__)
    log.info(f"Run dir : {run_dir}")
    log.info(f"Exp     : {cfg.exp_name}  |  Model: {cfg.method.model_name}  |  Dataset: {cfg.dataset.dataset_name}")

    # ── Build model & save artifacts ──────────────────────────────────────────
    model = build_model(cfg)

    OmegaConf.save(cfg, os.path.join(run_dir, 'config_merged.yaml'), resolve=True)
    with open(os.path.join(run_dir, 'model.txt'), 'w') as f:
        print(model, file=f)
    log.info("Saved: config_merged.yaml, model.txt")

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    train_set = build_dataset(cfg)
    val_set   = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices), 1)
    eval_batch_size  = max(cfg.method['eval_batch_size']  // len(cfg.devices), 1)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size,
        num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size,
        num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # ── WandB Logger ──────────────────────────────────────────────────────────
    if cfg.debug:
        logger = None
    else:
        wb         = cfg.wandb
        tags       = list(wb.tags) if wb.tags else []
        run_time   = Path(run_dir).name                                       
        auto_name  = f"{cfg.exp_name}_{run_time}"                             
        auto_group = f"{cfg.method.model_name}_{cfg.dataset.dataset_name}"    

        logger = WandbLogger(
            project  = wb.project,
            name     = wb.name   if wb.name   else auto_name,
            id       = wb.id     if wb.id     else None,
            resume   = wb.get("resume", None),
            group    = wb.group  if wb.group  else auto_group,
            tags     = tags,
            notes    = wb.notes  if wb.notes  else None,
            job_type = wb.job_type,
            save_dir = run_dir,
            config   = OmegaConf.to_container(cfg, resolve=True),
        )
        if rank_zero_only.rank == 0:
            logger.experiment.summary.update({
                "run_dir"          : run_dir,
                "train_samples"    : len(train_set),
                "val_samples"      : len(val_set),
                "train_batch_size" : train_batch_size * len(cfg.devices),
                "eval_batch_size"  : eval_batch_size  * len(cfg.devices),
                "num_gpus"         : len(cfg.devices),
                "model"            : cfg.method.model_name,
                **get_git_info(),
            })
            log.info(f"WandB   : {logger.experiment.url}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(run_dir, 'checkpoints')

    callbacks = [
        ModelCheckpoint(
            monitor='val/brier_fde',
            filename='best-epoch={epoch}-val_brier_fde={val/brier_fde:.4f}',
            save_top_k=1,
            mode='min',
            dirpath=ckpt_dir,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            filename='last',
            save_last=True,
            dirpath=ckpt_dir,
        ),
        LearningRateMonitor(logging_interval='step'),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
    ]

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs        = cfg.method.max_epochs,
        logger            = logger,
        devices           = 1 if cfg.debug else cfg.devices,
        gradient_clip_val = cfg.method.grad_clip_norm,
        accelerator       = "cpu" if cfg.debug else "gpu",
        strategy          = "auto" if cfg.debug else "ddp",
        callbacks         = callbacks,
        # profiler          = "simple",
    )

    if cfg.ckpt_path is None and not cfg.debug:
        search_pattern = os.path.join(cfg.log_dir, cfg.exp_name, '*', 'checkpoints', '*.ckpt')
        cfg.ckpt_path  = find_latest_checkpoint(search_pattern)
        if cfg.ckpt_path:
            log.info(f"Auto-resume: {cfg.ckpt_path}")

    trainer.fit(
        model             = model,
        train_dataloaders = train_loader,
        val_dataloaders   = val_loader,
        ckpt_path         = cfg.ckpt_path,
    )


if __name__ == '__main__':
    train()
