import math

import torch
from torchinfo import summary
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
import ray
from ray import tune, air
# from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback


# constants
N_SAMPLES = 50_000
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

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(n_features, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 1)

        # example input array, required for TensorBoard Logger with
        # parameter log_graph=True
        self.example_input_array = torch.rand(1, n_features)

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
        # print(f">>>>{os.getpid()} entering training_step, epoch {self.current_epoch}, batch_idx: {batch_idx}, batch_size: {train_batch[0].shape}")
        x, y = train_batch
        logits = self.forward(x)
        loss = self.rmse_loss(logits, y)

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        # print(f">>>>{os.getpid()} entering validation_step, batch_idx: {batch_idx}, batch_size: {val_batch[0].shape}")
        x, y = val_batch
        logits = self.forward(x)
        loss = self.rmse_loss(logits, y)
        self.val_loss_list.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.val_loss_list).mean()
        self.log("ptl/val_loss", avg_loss)

    def on_before_optimizer_step(self, optimizer):
            # example to inspect gradient information in tensorboard
            if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


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


###
# Display model summary
###
# model = RegressionModel(
#     config={"layer_1_size": 32, "layer_2_size": 64, "lr": 1e-4, },
#     n_features=N_FEATURES
# )
# print(model)
# print(summary(model, input_size=(1, N_FEATURES)))
# sys.exit(0)



###
# test trainng the LightningModule
###
# model = RegressionModel(
#     config={"layer_1_size": 32, "layer_2_size": 64, "lr": 1e-4, }, 
#     n_features=N_FEATURES
# )

# data_module = RegressionDataModule(data_fp="../data/data.parquet", batch_size=128)

# trainer = pl.Trainer(max_epochs=3, enable_progress_bar=False)
# trainer.fit(model, data_module)
# sys.exit(0)


###
# setup for ray tune
###

def train_regression_tune(config, num_epochs=10, num_cpus=1, num_gpus=0, data_fp=None):
    """
    Core training loop for tune run
    """
    print(f">>>>{os.getpid()} entering train_regression_tune with config: {config}")
    print(f">>>>{os.getpid()} has access to {os.cpu_count()} cpus")
    print(f">>>>{os.getpid()} entering torch num_threads {torch.get_num_threads()}")
    trial_id = ray.tune.get_trial_id()
    worker_id = ray.worker.global_worker.worker_id
    print(f">>>>{os.getpid()} trial_id: {trial_id}, worker_id: {worker_id.hex()}")

    start_time = time.time()
    torch.set_num_threads(num_cpus)
    print(f">>>>{os.getpid()} after setting torch num_threads {torch.get_num_threads()}")

    model = RegressionModel(config, n_features=N_FEATURES)

    data_module = RegressionDataModule(
        data_fp=data_fp, 
        batch_size=config["batch_size"],
        num_workers=num_cpus,
    )

    # define EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="ptl/val_loss",
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        # gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            log_graph=True,
            save_dir=os.getcwd(), name="", version="."
        ),
        enable_progress_bar=False,
        callbacks=[
            early_stop_callback,
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
    )
    # following does not appear in stdout, not sure why
    print(f">>>>{os.getpid()} exiting train_regression_tune after {time.time() - start_time} seconds\n")
    # sys.stdout.flush()  # does not work


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
        max_report_frequency=15,
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "time_total_s", "training_iteration"])

    # setup for train function with parameters fir tune
    train_fn_with_parameters = tune.with_parameters(
        train_regression_tune,
        num_epochs=num_epochs,
        num_cpus=cpus_per_trial,
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
    shutil.rmtree("lightning_logs", ignore_errors=True, onerror=None)

    # run the hyperparameter tuning
    tune_regression_asha(
        num_epochs=10,
        num_samples=5,
        cpus_per_trial=2,
        data_fp="/workspaces/devcontainer_testbed/data/data.parquet"
    )

    # all done
    print("Done!")
