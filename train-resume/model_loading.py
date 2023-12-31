import os
import shutil

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
        self.save_hyperparameters()
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
        loss = F.mse_loss(y_hat, y.reshape(-1, 1))
        self.log("ptl/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.reshape(-1, 1))
        self.log("ptl/val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.reshape(-1, 1))
        self.log("ptl/test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_loss_list.append(loss.item())

    def on_test_end(self) -> None:
        overall_test_loss = torch.stack([torch.tensor(x) for x in self.test_loss_list]).mean()
        self.overall_test_loss = overall_test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

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

if __name__ == "__main__":

    # load data
    train_df = pd.read_parquet(os.path.join("./data", "train_data.parquet"))
    test_df = pd.read_parquet(os.path.join("./data", "test_data.parquet"))

    x = torch.tensor(np.random.rand(7, 100), dtype=torch.float32)
    print(x.shape)

    shutil.rmtree("lightning_logs/testbed", ignore_errors=True)

    # define data module
    data_module = RegressionDataModule(train_df, test_df)

    # define model
    model = RegressionModel(input_dim=100, output_dim=1)

    # define trainer
    trainer = pl.Trainer(
        max_epochs=3,
        logger=pl.loggers.TensorBoardLogger(
            "lightning_logs/", 
            name="testbed", 
            version="bed1"
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="ptl/val_loss", save_last=True, mode="min"),
        ],
    )    

    # train model
    trainer.fit(model, data_module)

print(">>>>>>PREDICTIONS WITH ORIGINAL MODEL<<<<<")
with torch.no_grad():
    print(trainer.model(x))    

# load trained model
print(">>>>>>MODEL LOADING<<<<<<")
model2 = RegressionModel.load_from_checkpoint(
    "lightning_logs/testbed/bed1/checkpoints/last.ckpt",
)

# make predictions  
print(type(model2))

print(">>>>>>predictions with torch.no_grad()<<<<<<")
with torch.no_grad():   
    print(model2(x))

# alternative way to make predictions
print(">>>>>>predictions with eval()/freeze()<<<<<<")
model2.eval()
model2.freeze()
print(model2(x))


print(">>>>>>predictions with dataloader<<<<<<")
dl = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)
for batch in dl:
    print(model2(batch[0]))