# src/models/regression_module.py

from typing import Any, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric, MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


class RegressionLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        criterion: Optional[torch.nn.Module] = None,  # <--- 改进点: 损失函数配置化
        compile: bool = False,
    ) -> None:
        super().__init__()
        # <--- 改进点: 忽略复杂对象，不再需要datamodule ---
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = net
        # <--- 改进点: 使用传入的criterion，或提供一个默认值 ---
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()

        metrics = MetricCollection(
            {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        preds = self.forward(x).squeeze(-1)
        loss = self.criterion(preds, y.float())
        return loss, preds, y

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_metrics(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_metrics(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        # <--- 改进点: 逻辑简化，只返回预测值。回调函数会处理保存 ---
        if isinstance(batch, (tuple, list)) and len(batch) == 2:  # Labeled test data
            loss, preds, targets = self.model_step(batch)
            self.test_loss(loss)
            self.test_metrics(preds, targets)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
            return preds
        else:  # Unlabeled test data
            preds = self.forward(batch).squeeze(-1)
            return preds

    # <--- 改进点: 移除了 on_test_start, on_test_epoch_end, set_test_ids ---

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        return {"optimizer": optimizer}
