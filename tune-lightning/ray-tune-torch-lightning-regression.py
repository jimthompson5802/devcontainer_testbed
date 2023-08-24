import math

import torch
import pytorch_lightning as pl
from filelock import FileLock
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import os
import sys
import shutil
import time
from sklearn.datasets import make_regression
import pandas as pd

from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback


# constants
N_SAMPLES = 500000
N_FEATURES = 100

# create synthetic regression dataset
X, y = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, noise=0.1, random_state=1)

# convert to pandas dataframe
df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(N_FEATURES)])
df["y"] = y


# print size of dataframe
print(f"size of dataframe: {df.memory_usage(deep=True).sum() / (1024*1024)} MB")
print(df.head())

# save dataframe to paraquet
df.to_parquet("../data/data.parquet")


# class for custom pytorch dataset
class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df.drop(["y"], axis=1).values.astype("float32")
        self.target = df["y"].values.astype("float32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.target[idx]


class RegressionModel(pl.LightningModule):

    def __init__(self, config, n_features: int=None) -> None:
        super().__init__()

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(n_features, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 1)

        self.val_loss_list = []

    def forward(self, x) -> torch.Tensor:

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)

        return x

    def rmse_loss(self, logits, labels) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(logits, labels.reshape(-1, 1)))


    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        x, y = train_batch
        logits = self.forward(x)
        loss = self.rmse_loss(logits, y)

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        x, y = val_batch
        logits = self.forward(x)
        loss = self.rmse_loss(logits, y)
        self.val_loss_list.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.val_loss_list).mean()
        self.log("ptl/val_loss", avg_loss)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Define the LightningDataModule
class RegressionDataModule(pl.LightningDataModule):
    def __init__(self, data_fp: str=None, batch_size: int=64) -> None:
        super().__init__()
        self.data_fp = data_fp
        self.batch_size = batch_size

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
        return DataLoader(self.train_data, batch_size=int(self.batch_size))

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=int(self.batch_size))


###
# test trainng the LightningModule
###
# model = RegressionModel(
#     config={"layer_1_size": 32, "layer_2_size": 64, "lr": 1e-4, "batch_size": 32}, 
#     n_features=N_FEATURES
# )

# data_module = RegressionDataModule(data_fp="../data/data.parquet", batch_size=32)
# data_module.prepare_data()

# trainer = pl.Trainer(max_epochs=3, enable_progress_bar=True)
# trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())



###
# setup for ray tune
###

def train_regression_tune(config, num_epochs=10, num_gpus=0, data_fp=None):
    """
    Core training loop for tune run
    """
    print(f">>>>{os.getpid()} entering train_regression_tune with config: {config}")
    print(f">>>>{os.getpid()} has access to {os.cpu_count()} cpus")

    start_time = time.time()

    model = RegressionModel(config, n_features=N_FEATURES)

    data_module = RegressionDataModule(data_fp=data_fp, batch_size=32)
    # data_module.prepare_data()

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        # gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=os.getcwd(), name="", version="."),
        enable_progress_bar=False,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                },
                on="validation_end")
        ]
    )

    print(f">>>>{os.getpid()} starting trainer.fit")
    trainer.fit(
        model, 
        data_module,
        # train_dataloaders=data_module.train_dataloader(), 
        # val_dataloaders=data_module.val_dataloader()
    )
    print(f">>>>{os.getpid()} exiting train_regression_tune after {time.time() - start_time} seconds")


def tune_regression_asha(num_samples=10, num_epochs=10, cpus_per_trial=1, gpus_per_trial=0, data_fp=None):
    """
    Setup for the hyperparameter tuning with ASHA
    """
    start_time = time.time()

    config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }
    print(f">>>>{os.getpid()} entering train_regression_asha with config: {config}")

    # define the scheduler and reporter
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    # setup for train function with parameters fir tune
    train_fn_with_parameters = tune.with_parameters(
        train_regression_tune,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
        data_fp=data_fp
    )
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    
    # Configure tune with all the needed components
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="tune_regression_asha",
            progress_reporter=reporter,
            log_to_file=True,
            local_dir="ray_results",
        ),
        param_space=config,
    )

    # run the hyperparameter tuning processes
    results = tuner.fit()

    print(f"resources for each trial: {resources_per_trial}")
    print(
        f"Best hyperparameters found were:\n"
        f"  loss metric: {results.get_best_result().metrics['loss']}\n"
        f"  config: {results.get_best_result().config}"
    )
    print(f"exiting train_regression_asha after {time.time() - start_time} seconds")

if __name__ == "__main__":
    # clear out ray results directory
    shutil.rmtree("ray_results", ignore_errors=True, onerror=None)

    # run the hyperparameter tuning
    tune_regression_asha(
        num_samples=15,
        cpus_per_trial=2,
        data_fp="/workspaces/devcontainer_testbed/data/data.parquet"
    )

    # all done
    print("Done!")
