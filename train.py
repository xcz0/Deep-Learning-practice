# src/train.py
from typing import List, Optional
import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback
from src.utils import pylogger


# (可选) 从 .env 文件加载环境变量，例如 API 密钥
dotenv.load_dotenv(override=True)

# (可选) 导入自定义的工具函数，例如用于日志记录
log = pylogger.get_pylogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def train(cfg: DictConfig) -> Optional[float]:
    """
    使用 Hydra 配置来训练 PyTorch Lightning 模型。

    Args:
        cfg (DictConfig): 由 Hydra 组合和实例化的配置对象。

    Returns:
        Optional[float]: 由优化目标指定的指标值，用于超参数优化。
    """

    # 1. 打印和记录配置
    log.info("Starting training with the following configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # 2. 使用 hydra.utils.instantiate 实例化所有组件
    #    --- 数据模块 ---
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # --- 实例化 Task (LightningModule) ---
    # !! 修改点：不再直接实例化model，而是实例化task !!
    # task模块内部会负责实例化真正的模型（如ResNet）
    log.info(f"Instantiating task <{cfg.task._target_}>")
    # 动态将datamodule的属性注入到task的配置中
    # 这是实现数据和模型解耦的关键！
    task_cfg = OmegaConf.merge(
        cfg.task,
        {
            "model": {
                "num_classes": datamodule.num_classes,
                # "input_dims": datamodule.input_dims, # 其他可能需要的属性
            }
        },
    )
    model: LightningModule = hydra.utils.instantiate(task_cfg)

    #    --- 回调函数 (Callbacks) ---
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    #    --- 日志记录器 (Logger) ---
    logger: List[Logger] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # 3. 实例化 Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    # 4. 开始训练
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # 5. 开始测试
    if cfg.get("test_after_training"):
        log.info("Starting testing!")
        # 使用最佳模型检查点进行测试
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # 6. 完成
    best_model_path = None
    if hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback:
        best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", None)

    log.info(f"Training finished! Best model path: {best_model_path}")

    # (可选) 返回优化的目标指标
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric].item()

    return None


if __name__ == "__main__":
    train()
