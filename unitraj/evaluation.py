import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import hydra
torch.set_float32_matmul_precision('medium')

from datasets import build_dataset
from models import build_model
from utils.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluation(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg = OmegaConf.merge(cfg, cfg.dataset)
    cfg['eval'] = True

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = HydraConfig.get().runtime.output_dir

    # ── File logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(run_dir, 'eval.log')),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger(__name__)
    log.info(f"Run dir : {run_dir}")
    log.info(f"Exp     : {cfg.exp_name}  |  Model: {cfg.method.model_name}  |  Dataset: {cfg.dataset.dataset_name}")
    log.info(f"Ckpt    : {cfg.ckpt_path}")

    # ── Build model & save config ─────────────────────────────────────────────
    model = build_model(cfg)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config_merged.yaml'), resolve=True)

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    val_set         = build_dataset(cfg, val=True)
    eval_batch_size = cfg.method['eval_batch_size']

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size,
        num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn)

    # ── WandB Logger ──────────────────────────────────────────────────────────
    if cfg.debug:
        logger = None
    else:
        wb         = cfg.wandb
        tags       = list(wb.tags) if wb.tags else []
        if "eval" not in tags:
            tags.append("eval")
        run_time   = Path(run_dir).name
        auto_name  = f"eval_{cfg.exp_name}_{run_time}"                        # "eval_DeMo-v1_20240228-120000"
        auto_group = f"{cfg.method.model_name}_{cfg.dataset.dataset_name}"

        logger = WandbLogger(
            project  = wb.project,
            name     = wb.name   if wb.name   else auto_name,
            group    = wb.group  if wb.group  else auto_group,
            tags     = tags,
            notes    = wb.notes  if wb.notes  else None,
            job_type = "eval",
            save_dir = run_dir,
            config   = OmegaConf.to_container(cfg, resolve=True),
        )
        logger.experiment.summary.update({
            "run_dir"     : run_dir,
            "val_samples" : len(val_set),
            "model"       : cfg.method.model_name,
            "ckpt_path"   : cfg.ckpt_path,
        })
        log.info(f"WandB   : {logger.experiment.url}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        inference_mode = True,
        logger         = logger,
        devices        = 1,
        accelerator    = "cpu" if cfg.debug else "gpu",
        # profiler       = "simple",
    )

    results = trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    results_path = os.path.join(run_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Results → {results_path}")
    if results:
        console = Console()
        table = Table(title=f"Eval Results  [{cfg.method.model_name} / {cfg.dataset.dataset_name}]",
                      show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim", min_width=40)
        table.add_column("Value", justify="right", style="bold green")
        for k, v in sorted(results[0].items()):
            table.add_row(k, f"{v:.4f}")
        console.print(table)


if __name__ == '__main__':
    evaluation()
