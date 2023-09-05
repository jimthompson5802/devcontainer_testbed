import math

import torch
from torch import nn, functional as F
from torchinfo import summary
from torchvision.datasets import MNIST
from torchvision import transforms
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
from typing import Optional

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

# config pprint.pprint
pp = pprint.PrettyPrinter(indent=4)

# constants
N_FEATURES = 100
TENSORBOARD_LOG_FREQUENCY = 25

# retrieve run-time config
with open("config_mnist.yaml", "r") as f:
    config = yaml.safe_load(f)

pp.pprint(config)

DATA_DIR = config["data_dir"]
RAY_RESULTS_DIR = config["ray_results_dir"]
LIGHTNING_LOGS_DIR = config["lightning_logs_dir"]

# # class for custom pytorch dataset
# class CustomDataset(Dataset):
#     def __init__(self, df: pd.DataFrame) -> None:
#         self.data = df.drop(["y"], axis=1).values.astype("float32")
#         self.target = df["y"].values.astype("float32")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx: int):
#         return self.data[idx], self.target[idx]


class TheModel(pl.LightningModule):

    def __init__(self, config, ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(config["dropout1"])  
        self.dropout2 = nn.Dropout(config["dropout2"])
        self.fc1 = nn.Linear(9216, config["fc2_size"])
        self.fc2 = nn.Linear(config["fc2_size"], 10)

        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        self.val_loss_list = []
        self.val_accuracy_list = []

        # example input array, required for TensorBoard Logger with
        # parameter log_graph=True
        self.example_input_array = torch.rand(1, 1, 28, 28)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        self.logger.experiment.add_histogram(
            tag="actv/conv1",
            values=x, 
            global_step=self.trainer.global_step
        )
  
        x = self.conv2(x)
        x = F.relu(x)
        self.logger.experiment.add_histogram(
            tag="actv/conv2",
            values=x, 
            global_step=self.trainer.global_step
        )

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def loss_function(self, logits, labels) -> torch.Tensor:
        return F.nll_loss(logits, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        # print(f">>>>{os.getpid()} entering training_step, epoch {self.current_epoch}, batch_idx: {batch_idx}, batch_size: {train_batch[0].shape}")
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss_function(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        # print(f">>>>{os.getpid()} entering validation_step, batch_idx: {batch_idx}, batch_size: {val_batch[0].shape}")
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss_function(logits, y)
        accuracy = self.accuracy(logits, y)
        self.val_loss_list.append(loss)
        self.val_accuracy_list.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.val_loss_list).mean()
        avg_accuracy = torch.stack(self.val_accuracy_list).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_accuracy)

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
class TheDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_dir: str=None, 
            batch_size: int=64, 
            num_workers: int=1,
            train_sample_size: [Optional, int]=None,
    )-> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sample_size = train_sample_size

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )


    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            mnist = MNIST(
                self.data_dir, train=True, download=True, transform=self.transform
            )
            self.train_data, self.val_data = random_split(mnist, [55000, 5000])

            self.mnist_test = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )

            if self.train_sample_size is not None:
                # get size of train dataset
                train_size = len(self.train_data)

                # select random 100 random numbers in the range 0 to train_size
                train_idx = torch.randint(high=train_size, size=(self.train_sample_size,)).tolist()

                # setup subset random sampler with the train_idx
                self.train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            else:
                self.train_sampler = None

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=int(self.batch_size), 
            num_workers=self.num_workers,
            sampler=self.train_sampler,
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
default_config = {
    "dropout1": 0.25, 
    "dropout2": 0.1,
    "fc2_size": 10,
    "lr": 1e-4, 
    "batch_size": 64,
}
model = TheModel(config=default_config)
print(model)
# print(summary(model, input_size=(1, 1, 28, 28)))
# sys.exit(0)



###
# test trainng the LightningModule
###
# model = TheModel(config=default_config)
# data_module = TheDataModule(data_dir=DATA_DIR, batch_size=128, train_sample_size=10000)
# trainer = pl.Trainer(max_epochs=3, enable_progress_bar=True)
# trainer.fit(model, data_module)
# sys.exit(0)


###
# setup for ray tune
###

def train_tune(config, num_epochs=10, num_cpus=1, num_gpus=0, data_dir=None, train_sample_size=None):
    """
    Core training loop for tune run
    """
    print(f">>>>{os.getpid()} entering train_tune with config: {config}")
    print(f">>>>{os.getpid()} has access to {os.cpu_count()} cpus")
    print(f">>>>{os.getpid()} entering torch num_threads {torch.get_num_threads()}")
    trial_id = ray.tune.get_trial_id()
    worker_id = ray.worker.global_worker.worker_id
    print(f">>>>{os.getpid()} trial_id: {trial_id}, worker_id: {worker_id.hex()}")

    start_time = time.time()
    torch.set_num_threads(num_cpus)
    print(f">>>>{os.getpid()} after setting torch num_threads {torch.get_num_threads()}")

    model = TheModel(config,)

    data_module = TheDataModule(
        data_dir=DATA_DIR, 
        batch_size=config["batch_size"],
        num_workers=num_cpus,
        train_sample_size=train_sample_size,
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
                    "accuracy": "ptl/val_accuracy",
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


def tune_asha(num_samples=10, num_epochs=10, cpus_per_trial=1, gpus_per_trial=0, data_dir=None, train_sample_size=None):
    """
    Setup for the hyperparameter tuning with ASHA
    """
    start_time = time.time()

    config = {
        "dropout1": tune.choice([0.10, 0.15, 0.25,]), 
        "dropout2": tune.choice([0.05, 0.1, 0.15,]),
        "fc2_size": tune.choice([512, 256, 128, 64, ]),
        "lr":tune.loguniform(1e-4, 1e-1), 
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
        parameter_columns=["dropout1", "dropout2", "fc2_size", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "time_total_s", "training_iteration"])

    # setup for train function with parameters fir tune
    train_fn_with_parameters = tune.with_parameters(
        train_tune,
        num_epochs=num_epochs,
        num_cpus=cpus_per_trial,
        num_gpus=gpus_per_trial,
        data_dir=data_dir,
        train_sample_size=train_sample_size,
    )
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    
    # Configure tune with all the needed components
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="tune_mnist_asha",
            progress_reporter=reporter,
            log_to_file=True,
            local_dir=RAY_RESULTS_DIR,
        ),
        param_space=config,
    )

    # run the hyperparameter tuning processes
    results = tuner.fit()

    print(f"resources for each trial: {resources_per_trial}")
    print(
        f"Best hyperparameters found were:\n"
        f"  accuracy metric: {results.get_best_result().metrics['accuracy']}\n"
        f"  loss metric: {results.get_best_result().metrics['loss']}\n"
        f"  config: {results.get_best_result().config}"
    )
    print(f"exiting train_regression_asha after {time.time() - start_time} seconds")

if __name__ == "__main__":
    # clear out ray results directory
    shutil.rmtree(RAY_RESULTS_DIR, ignore_errors=True, onerror=None)
    shutil.rmtree(LIGHTNING_LOGS_DIR, ignore_errors=True, onerror=None)

    # run the hyperparameter tuning
    tune_asha(
        num_epochs=10,
        num_samples=5,
        cpus_per_trial=2,
        data_dir=DATA_DIR,
        train_sample_size=1000,
    )

    # all done
    print("Done!")
