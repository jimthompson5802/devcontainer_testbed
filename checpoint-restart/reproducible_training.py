# load libraries
import os
import shutil
import sys

import pandas as pd
import numpy as np


# define Torch Lightning Module for regression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import pytorch_lightning as pl


class RegressionModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.test_loss_list = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("ptl/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("ptl/val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("ptl/test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_loss_list.append(loss.item())

    def on_test_end(self) -> None:
        overall_test_loss = torch.stack([torch.tensor(x) for x in self.test_loss_list]).mean()
        self.overall_test_loss = overall_test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

# define data module for training and test data
from torch.utils.data import DataLoader, TensorDataset


class RegressionDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, batch_size=64):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size

    def setup(self, stage=None):

        if stage == "fit":
            # Split train into train and validation
            train_size = int(0.8 * len(self.train_df))
            val_size = len(self.train_df) - train_size
            self.train_df, self.val_df = random_split(self.train_df, [train_size, val_size])

            self.train_X = torch.tensor(self.train_df.dataset.drop("y", axis=1).values, dtype=torch.float32)
            self.train_y = torch.tensor(self.train_df.dataset["y"].values, dtype=torch.float32)

            self.val_X = torch.tensor(self.val_df.dataset.drop("y", axis=1).values, dtype=torch.float32)
            self.val_y = torch.tensor(self.val_df.dataset["y"].values, dtype=torch.float32)
    
        if stage == "test":
            # Test data set
            self.test_X = torch.tensor(self.test_df.drop("y", axis=1).values, dtype=torch.float32)
            self.test_y = torch.tensor(self.test_df["y"].values, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.train_X, self.train_y), 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.val_X, self.val_y), batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(TensorDataset(self.test_X, self.test_y), batch_size=self.batch_size, shuffle=False)

def do_a_single_run(train_df, test_df, seed, logger_version=None):
    print(f">>>>single run: {logger_version}")
    # set seeds to ensure reproducibility
    pl.seed_everything(seed)

    # define data module
    data_module = RegressionDataModule(train_df, test_df)

    # define model
    model = RegressionModel(input_dim=100, output_dim=1)

    # define trainer
    trainer = pl.Trainer(
        max_epochs=15,
        deterministic=True,
        logger=pl.loggers.TensorBoardLogger(
            "lightning_logs/", 
            name="reproducibility", 
            version=logger_version
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="ptl/val_loss", save_last=True, mode="min"),
        ],
    )    

    # train model
    trainer.fit(model, data_module)

    # test model
    trainer.test(model, data_module)
    run_test_loss = model.overall_test_loss

    return run_test_loss, trainer.model

def do_a_resume_run(train_df, test_df, seed, logger_version=None):
    print(f">>>>start of resume run initial training: {logger_version}")
    # set seeds to ensure reproducibility
    pl.seed_everything(seed)

    # define data module
    data_module = RegressionDataModule(train_df, test_df)

    # define model
    model = RegressionModel(input_dim=100, output_dim=1)

    # define trainer for initial run
    trainer = pl.Trainer(
        max_epochs=10,
        deterministic=True,
        logger=pl.loggers.TensorBoardLogger(
            "lightning_logs/", 
            name="reproducibility", 
            version=logger_version
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="ptl/val_loss", save_last=True, mode="min"),
        ],
    )    

    # train model
    trainer.fit(model, data_module)

    # trainer.save_checkpoint("lightning_logs/reproducibility/"+ logger_version + "/checkpoints/model_save.ckpt")
    ckpt = torch.load("lightning_logs/reproducibility/"+ logger_version + "/checkpoints/last.ckpt")


    # define trainer for resume training run
    print(f">>>>start of resume training run: {logger_version}")
    # define model
    model_resume = RegressionModel(100,1)

    trainer_resume = pl.Trainer(
        max_epochs=15,
        deterministic=True,
        logger=pl.loggers.TensorBoardLogger(
            "lightning_logs/", 
            name="reproducibility", 
            version=logger_version
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="ptl/val_loss", save_last=True, mode="min"),
        ],
    )    

    # define data module
    data_module_resume = RegressionDataModule(train_df, test_df)


    # train model
    trainer_resume.fit(
        model_resume, 
        data_module_resume,
        ckpt_path="lightning_logs/reproducibility/"+ logger_version + "/checkpoints/last.ckpt"
    )


    # test model
    trainer_resume.test(model_resume, data_module_resume)
    run_test_loss = model_resume.overall_test_loss

    return run_test_loss, trainer_resume.model

def compare_model_parameters(model1: pl.LightningModule, model2: pl.LightningModule)-> bool :
    return np.all(
        [torch.allclose(param1, param2) for param1, param2 in zip(model1.parameters(), model2.parameters())]
    )


if __name__ == "__main__":
    # load data
    train_df = pd.read_parquet(os.path.join("./data", "train_data.parquet"))
    test_df = pd.read_parquet(os.path.join("./data", "test_data.parquet"))

    shutil.rmtree("lightning_logs/", ignore_errors=True)    

    run1_test_loss, model1 = do_a_single_run(train_df, test_df, 1919, logger_version="run1")
    run2_test_loss, model2 = do_a_single_run(train_df, test_df, 1919, logger_version="run2")
    run3_test_loss, model3 = do_a_resume_run(train_df, test_df, 1919, logger_version="run3")
    print(
        f"run1_test_loss: {run1_test_loss:0.4f}\n"
        f"run2_test_loss: {run2_test_loss:0.4f}\n"
        f"run3_test_loss: {run3_test_loss:0.4f}\n"
    )
  
    # check if model parameters are the same
    print(f"run1 and run2 model parameters are the same: {compare_model_parameters(model1, model2)}")
    print(f"run1 and run3 model parameters are the same: {compare_model_parameters(model1, model3)}")