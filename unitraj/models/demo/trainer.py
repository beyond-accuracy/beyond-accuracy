"""DeMo Trainer for UniTraj.

Integrates the original DeMo (Denoising Motion) into UniTraj BaseModel.

Loss components:
- dense_reg_loss: state query trajectory regression (smooth L1)
- agent_reg_loss: mode query trajectory regression (smooth L1)
- agent_cls_loss: mode classification (cross entropy with label smoothing)
- new_agent_reg_loss: refined trajectory regression (smooth L1)
- new_pi_reg_loss: refined mode classification (cross entropy)
- others_reg_loss: other agents trajectory regression (smooth L1)
- laplace_loss / laplace_loss_new: Laplace NLL loss
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ModelForecast
from .optim import WarmupCosLR
from .laplace_nll_loss import LaplaceNLLLoss
from unitraj.models.base_model.base_model import BaseModel


class DeMoTrainer(BaseModel):
    def __init__(
        self,
        config=None,
        # model architecture
        embed_dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path: float = 0.2,
        # training
        lr: float = 3e-4,
        warmup_epochs: int = 10,
        epochs: int = 64,
        weight_decay: float = 1e-4,
        pretrained_weights: str = None,
    ) -> None:
        super(DeMoTrainer, self).__init__(config=config)

        # Override defaults from UniTraj config if provided
        if config is not None:
            embed_dim          = config.get("embed_dim", embed_dim)
            num_heads          = config.get("tx_num_heads", num_heads)
            mlp_ratio          = config.get("mlp_ratio", mlp_ratio)
            qkv_bias           = config.get("qkv_bias", qkv_bias)
            drop_path          = config.get("drop_path", drop_path)
            lr                 = config.get("learning_rate", lr)
            warmup_epochs      = config.get("warmup_epochs", warmup_epochs)
            epochs             = config.get("max_epochs", epochs)
            weight_decay       = config.get("weight_decay", weight_decay)
            pretrained_weights = config.get("pretrained_weights", pretrained_weights)

        self.warmup_epochs = warmup_epochs
        self.epochs        = epochs
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.future_steps  = config.get("future_len", 60) if config else 60
        self.history_steps = config.get("past_len", 21) if config else 21
        self.save_hyperparameters()

        self.net = ModelForecast(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=self.future_steps,
        )

        self.laplace_loss = LaplaceNLLLoss()

        if pretrained_weights is not None:
            result = self.net.load_from_checkpoint(pretrained_weights)
            print(f"Pretrained weights loaded — missing: {len(result.missing_keys)}, "
                  f"unexpected: {len(result.unexpected_keys)}")

        total     = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"[DeMo] Total params: {total:,} | Trainable: {trainable:,}")

    # ── Data format conversion ────────────────────────────────────────────────

    def _get_demo_data(self, batch):
        """Get pre-converted DeMo data from collate, or convert on-the-fly."""
        if 'demo_data' in batch:
            return batch['demo_data']
        # Fallback for non-FMAEDataset loaders (e.g. LOO analysis)
        from unitraj.datasets.fmae_dataset import convert_to_demo_format
        device = next(self.parameters()).device
        demo_data = convert_to_demo_format(batch['input_dict'], self.config, device=device)
        # Ensure all tensors are on model device (guard against CPU roundtrip in convert)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in demo_data.items()}

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, batch):
        demo_data = self._get_demo_data(batch)
        return self.net(demo_data)

    def predict(self, batch):
        with torch.no_grad():
            demo_data = self._get_demo_data(batch)
            out = self.net(demo_data)

        # Use refined output if available
        y_hat = out["new_y_hat"] if out["new_y_hat"] is not None else out["y_hat"]
        pi    = out["new_pi"] if out["new_pi"] is not None else out["pi"]

        prediction = {
            'predicted_trajectory': y_hat[..., :2],
            'predicted_probability': torch.softmax(pi.double(), dim=-1),
        }
        return prediction

    # ── Loss computation ────────────────────────────────────────────────────

    def cal_loss(self, out, data):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)

        # gt
        y, y_others = data["target"][:, 0], data["target"][:, 1:]

        # loss for output of state query
        if dense_predict is not None:
            dense_reg_loss = F.smooth_l1_loss(dense_predict, y)
        else:
            dense_reg_loss = 0

        # loss for output of mode query
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)

        # loss for final output
        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0
        if new_pi is not None:
            new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_pi_reg_loss = 0

        # loss for other agents
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        # Laplace loss
        predictions = {}
        predictions['traj'] = y_hat
        predictions['scale'] = scal
        predictions['probs'] = pi
        laplace_loss = self.laplace_loss.compute(predictions, y)

        predictions['traj'] = new_y_hat
        predictions['scale'] = scal_new
        predictions['probs'] = new_pi
        laplace_loss_new = self.laplace_loss.compute(predictions, y)

        # total loss
        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + \
                new_agent_reg_loss + dense_reg_loss + new_pi_reg_loss
        loss = loss + laplace_loss + laplace_loss_new

        loss_dict = {
            "loss": loss.item(),
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
            "laplace_loss": laplace_loss.item(),
            "laplace_loss_new": laplace_loss_new.item(),
        }
        if new_y_hat is not None:
            loss_dict["reg_loss_refine"] = new_agent_reg_loss.item()
        if new_pi is not None:
            loss_dict["cls_loss_refine"] = new_pi_reg_loss.item()
        if dense_predict is not None:
            loss_dict["reg_loss_dense"] = dense_reg_loss.item()

        return loss, loss_dict

    # ── Training / Validation steps ──────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        demo_data = self._get_demo_data(batch)
        out = self.net(demo_data)
        loss, loss_dict = self.cal_loss(out, demo_data)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}", v,
                on_step=True, on_epoch=True,
                prog_bar=(k == "loss"), sync_dist=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)

        # Use refined output if available
        y_hat = out["new_y_hat"] if out["new_y_hat"] is not None else out["y_hat"]
        pi    = out["new_pi"] if out["new_pi"] is not None else out["pi"]

        prediction = {
            'predicted_trajectory': y_hat[..., :2],
            'predicted_probability': torch.softmax(pi.double(), dim=-1),
        }
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status='val')

    # ── Optimizer ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d,
                     nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                     nn.SyncBatchNorm, nn.LayerNorm, nn.Embedding)

        for mod_name, mod in self.named_modules():
            for par_name, par in mod.named_parameters():
                if not par.requires_grad:
                    continue
                full = f"{mod_name}.{par_name}" if mod_name else par_name
                if "bias" in par_name:
                    no_decay.add(full)
                elif "weight" in par_name:
                    if isinstance(mod, whitelist):
                        decay.add(full)
                    elif isinstance(mod, blacklist):
                        no_decay.add(full)
                else:
                    no_decay.add(full)

        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        assert len(decay & no_decay) == 0

        optim_groups = [
            {"params": [param_dict[n] for n in sorted(decay)],    "weight_decay": self.weight_decay},
            {"params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr)
        scheduler = WarmupCosLR(
            optimizer=optimizer, lr=self.lr, min_lr=1e-6,
            warmup_epochs=self.warmup_epochs, epochs=self.epochs)
        return [optimizer], [scheduler]
