# src/data/covid_datamodule.py

from typing import Optional, Dict
from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class COVID19Dataset(Dataset):
    def __init__(self, x, y=None):
        if not isinstance(x, torch.Tensor):
            self.x = torch.from_numpy(x).float()
        else:
            self.x = x.float()
        if y is not None:
            if not isinstance(y, torch.Tensor):
                self.y = torch.from_numpy(y).float()
            else:
                self.y = y.float()
        else:
            self.y = None

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]

    def __len__(self):
        return len(self.x)

    @property
    def feature_dim(self):
        return self.x.shape[1] if len(self.x.shape) > 1 else 1


class CovidDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        train_filename: str = "covid/covid.train.csv",  # <--- 改进点: 文件名配置化
        test_filename: str = "covid/covid.test.csv",  # <--- 改进点: 文件名配置化
        id_column: str = "id",  # <--- 改进点: 列名配置化
        target_column: str = "tested_positive",  # <--- 改进点: 列名配置化
        train_val_test_split: Optional[Dict[str, float]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        normalize_features: bool = True,
        random_seed: int = 42,
        **kwargs,  # 接受多余的参数
    ) -> None:
        super().__init__()
        if train_val_test_split is None:
            train_val_test_split = {"train": 0.8, "val": 0.2, "test": 0.0}
        total_split = sum(train_val_test_split.values())
        if not (0.99 <= total_split <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[COVID19Dataset] = None
        self.test_ids = None
        self.feature_stats = None

    @property
    def feature_dim(self) -> int:
        if not hasattr(self, "_feature_dim"):
            # 确保在访问维度前数据已准备好
            self.prepare_data()
            self.setup("fit")
        return self._feature_dim

    def setup(self, stage: Optional[str] = None) -> None:
        if (stage == "fit" and self.data_train is not None) or (
            stage == "test" and self.data_test is not None
        ):
            return

        data_dir = Path(self.hparams.data_dir)
        train_file = data_dir / self.hparams.train_filename
        test_file = data_dir / self.hparams.test_filename

        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        # <--- 改进点: 使用列名来分离特征、目标和ID，更加稳健 ---
        y_train = df_train[self.hparams.target_column].values.astype("float32")
        feature_cols = [
            col
            for col in df_train.columns
            if col not in [self.hparams.id_column, self.hparams.target_column]
        ]
        x_train = df_train[feature_cols].values.astype("float32")
        self._feature_dim = x_train.shape[1]

        if self.hparams.normalize_features:
            if self.feature_stats is None:
                self.feature_stats = {
                    "mean": x_train.mean(axis=0),
                    "std": x_train.std(axis=0) + 1e-8,
                }
            x_train = (x_train - self.feature_stats["mean"]) / self.feature_stats["std"]

        if stage in ("fit", None):
            full_train_dataset = COVID19Dataset(x_train, y_train)
            train_ratio = self.hparams.train_val_test_split["train"]
            val_ratio = self.hparams.train_val_test_split["val"]
            if (train_ratio + val_ratio) > 0:
                train_size = int(
                    len(full_train_dataset) * train_ratio / (train_ratio + val_ratio)
                )
                val_size = len(full_train_dataset) - train_size
                generator = torch.Generator().manual_seed(self.hparams.random_seed)
                self.data_train, self.data_val = random_split(
                    full_train_dataset, [train_size, val_size], generator=generator
                )

        if stage in ("test", None):
            self.test_ids = df_test[self.hparams.id_column].values
            test_feature_cols = [
                col for col in df_test.columns if col != self.hparams.id_column
            ]
            x_test = df_test[test_feature_cols].values.astype("float32")
            if self.hparams.normalize_features and self.feature_stats is not None:
                x_test = (x_test - self.feature_stats["mean"]) / self.feature_stats[
                    "std"
                ]
            self.data_test = COVID19Dataset(x_test)

    # train_dataloader, val_dataloader, test_dataloader等其他方法保持不变...
    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
            and self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last and shuffle,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        assert self.data_train is not None
        return self._create_dataloader(self.data_train, shuffle=True)

    def val_dataloader(self):
        assert self.data_val is not None
        return self._create_dataloader(self.data_val)

    def test_dataloader(self):
        assert self.data_test is not None
        return self._create_dataloader(self.data_test)
