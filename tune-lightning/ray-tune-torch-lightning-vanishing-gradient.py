import math

import torch
from torchinfo import summary
import pytorch_lightning as pl
from filelock import FileLock
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import os
import pprint
import sys
import shutil
import time
import yaml

import pandas as pd

import torch
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
# import ray
# from ray import tune, air
# # from ray.air import session
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
# from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
#     TuneReportCheckpointCallback

import pytorch_lightning as pl

# config pprint.pprint
pp = pprint.PrettyPrinter(indent=4)


# constants
N_SAMPLES = 10_000
N_FEATURES = 100
TENSORBOARD_LOG_FREQUENCY = 25

# retrieve run-time config
with open("config_regression.yaml", "r") as f:
    config = yaml.safe_load(f)

pp.pprint(config)

DATA_DIR = config["data_dir"]
RAY_RESULTS_DIR = config["ray_results_dir"]
LIGHTNING_LOGS_DIR = config["lightning_logs_dir"]


# class for custom pytorch dataset
class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df.drop(["y"], axis=1).values.astype("float32")
        self.target = df["y"].values.astype("float32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.target[idx]


class TheModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        num_layers =  config["num_layers"]
        output_size = config["output_size"]
        self.lr = config["lr"]

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Sigmoid())
        for _ in range(num_layers -1 ):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Sigmoid())
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.val_loss_list = []


    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
    
    def loss_function(self, logits, labels) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(logits, labels.reshape(-1, 1)))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        # print(f">>>>{os.getpid()} entering training_step, epoch {self.current_epoch}, batch_idx: {batch_idx}, batch_size: {train_batch[0].shape}")
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss_function(logits, y)

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        # print(f">>>>{os.getpid()} entering validation_step, batch_idx: {batch_idx}, batch_size: {val_batch[0].shape}")
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss_function(logits, y)
        self.val_loss_list.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.val_loss_list).mean()
        self.log("ptl/val_loss", avg_loss)

    def on_before_optimizer_step(self, optimizer):
            # example to inspect gradient information in tensorboard
            if self.trainer.global_step % TENSORBOARD_LOG_FREQUENCY == 0:  # don't make the tf file huge
                for k, v in self.named_parameters():
                    self.logger.experiment.add_histogram(
                        tag="raw/" + k, 
                        values=v, 
                        global_step=self.trainer.global_step
                    )
                    self.logger.experiment.add_histogram(
                        tag="grad/" + k, 
                        values=v.grad, 
                        global_step=self.trainer.global_step
                    )
                    self.logger.experiment.add_scalar(
                        "grad/" + k + "_norm", 
                        torch.linalg.vector_norm(v.grad).item(), 
                        global_step=self.trainer.global_step
                    )


# Define the LightningDataModule
class RegressionDataModule(pl.LightningDataModule):
    def __init__(self, data_fp: str=None, batch_size: int=64, num_workers: int=1) -> None:
        super().__init__()
        self.data_fp = data_fp
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        ###
        # Need to confirm this is the place to do all the data pre-processing
        # Also think about pre-processing in driver and store as Ray data object.
        ###
        df = CustomDataset(pd.read_parquet(self.data_fp))

        # split dataset into train, validation
        self.train_data, self.val_data = random_split(
            df, [int(len(df)*0.8), int(len(df)*0.2)])
        
    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=int(self.batch_size), 
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=int(self.batch_size), 
            num_workers=self.num_workers,
        )


if __name__ == "__main__":

    # clear out ray results directory
    shutil.rmtree(LIGHTNING_LOGS_DIR, ignore_errors=True, onerror=None)

    pl.seed_everything(13)

    ###
    # test trainng the LightningModule
    ###
    model = TheModel(
        config={
            "input_size": 100,
            "hidden_size": 32,
            "num_layers": 10,  # This makes the network very deep
            "output_size": 1,           
            "lr": 1e-4, 
        }, 
    )

    data_module = RegressionDataModule(
        data_fp=os.path.join(DATA_DIR, "data.parquet"), 
        batch_size=128
    )

    # define EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="ptl/val_loss",
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=20,
        # If fractional GPUs passed in, convert to int.
        # gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            log_graph=True,
            save_dir="lightning_logs", name="vanishing_gradient", version="bed1"
        ),
        enable_progress_bar=True,
        callbacks=[
            early_stop_callback,
        ]
    )

    trainer.fit(model, data_module)

    # all done
    print("Done!")
