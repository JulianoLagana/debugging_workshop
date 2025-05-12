import os
from typing import Optional, Tuple

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


class CSVDataset(Dataset):
    def __init__(self, path: str) -> None:
        df = pd.read_csv(path)
        self.X = df.drop(columns=["class_1"]).values.astype("float32")
        self.y = df["class_1"].values.astype("int64")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class CSVDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "splits", batch_size: int = 64) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = CSVDataset(os.path.join(self.data_dir, "train.csv"))
        self.val_ds = CSVDataset(os.path.join(self.data_dir, "val.csv"))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size)


class LitClassifier(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int = 64, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(1)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        acc = ((torch.sigmoid(logits) > 0.5) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        acc = ((torch.sigmoid(logits) > 0.5) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main() -> None:
    MyCLI(
        model_class=LitClassifier,
        datamodule_class=CSVDataModule,
        save_config_callback=None,
        seed_everything_default=1337,
    )


class MyCLI(LightningCLI):
    def configure_trainer(self) -> None:
        self.trainer.logger = self.logger

    def configure_model(self) -> None:
        input_dim = self.datamodule.train_ds.X.shape[1]
        self.model: LitClassifier = LitClassifier(
            input_dim=input_dim,
            hidden_dim=self.model.hparams.hidden_dim,
            lr=self.model.hparams.lr,
        )

    def after_fit(self) -> None:
        if self.trainer.max_epochs == 0:
            # save untrained model if necessary
            os.makedirs("models", exist_ok=True)
            self.trainer.save_checkpoint("models/trained-model.ckpt")
        else:
            plot_training_curves(self.trainer.logger)


def plot_training_curves(logger: CSVLogger) -> None:
    metrics_path = os.path.join(logger.log_dir, "metrics.csv")
    metrics_df = pd.read_csv(metrics_path)

    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

    plot_result(metrics_df, "train", "loss", ax[0], color="C0", smooth_window=20)
    plot_result(metrics_df, "val", "loss", ax[0], color="C1")
    configure_axis(ax[0], "Loss")

    plot_result(metrics_df, "train", "acc", ax[1], color="C0", smooth_window=20)
    plot_result(metrics_df, "val", "acc", ax[1], color="C1")
    configure_axis(ax[1], "Accuracy")

    plt.show()


def configure_axis(ax: plt.axis, ylabel: str) -> None:
    ax.set_xlabel("Gradient step")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()


def plot_result(
    df: pd.DataFrame, set: str, metric: str, ax: plt.axis, color: str, smooth_window: int = 0
) -> None:
    column_name = f"{set}_{metric}"
    nan_mask = df[column_name].isna()
    df = df[~nan_mask]

    y = df[column_name]
    if smooth_window > 1:
        y_smoothed = y.rolling(window=smooth_window, min_periods=1).mean()
        ax.plot(df["step"], y, c=color, alpha=0.5)
        ax.plot(df["step"], y_smoothed, label=f"{set} {metric}", c=color)
    else:
        ax.plot(df["step"], y, label=f"{set} {metric}", c=color)


if __name__ == "__main__":
    main()
