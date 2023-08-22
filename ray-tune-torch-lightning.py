import os
import torch
import tempfile
import pytorch_lightning as pl
import torch.nn.functional as F
from filelock import FileLock
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

from ray.train.lightning import LightningTrainer, LightningConfigBuilder


class MNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super(MNISTClassifier, self).__init__()
        self.accuracy = Accuracy("multiclass", num_classes=10)
        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)
        self.eval_loss = []
        self.eval_accuracy = []

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.data_dir = tempfile.mkdtemp()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            mnist = MNIST(
                self.data_dir, train=True, download=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])

            self.mnist_test = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)


default_config = {
    "layer_1_size": 128,
    "layer_2_size": 256,
    "lr": 1e-3,
}

from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining# The maximum training epochs
num_epochs = 5

# Number of sampls from parameter space
num_samples = 10

accelerator = "cpu" #"gpu"

dm = MNISTDataModule(batch_size=64)
logger = TensorBoardLogger(save_dir=os.getcwd(), name="tune-ptl-example", version=".")

# Static configs that does not change across trials
static_lightning_config = (
    LightningConfigBuilder()
    .module(cls=MNISTClassifier)
    .trainer(max_epochs=num_epochs, accelerator=accelerator, logger=logger)
    .fit_params(datamodule=dm)
    .checkpointing(monitor="ptl/val_accuracy", save_top_k=2, mode="max")
    .build()
)

# Searchable configs across different trials
searchable_lightning_config = (
    LightningConfigBuilder()
    .module(config={
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
    })
    .build()
)

# Make sure to also define an AIR CheckpointConfig here
# to properly save checkpoints in AIR format.
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="ptl/val_accuracy",
        checkpoint_score_order="max",
    ),
)

scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

scaling_config = ScalingConfig(
    num_workers=3, use_gpu=False, resources_per_worker={"CPU": 1, "GPU": 0}
)

# Define a base LightningTrainer without hyper-parameters for Tuner
lightning_trainer = LightningTrainer(
    lightning_config=static_lightning_config,
    scaling_config=scaling_config,
    run_config=run_config,
)

def tune_mnist_asha(num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": searchable_lightning_config},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        run_config=air.RunConfig(
            name="tune_mnist_asha",
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="ptl/val_accuracy", mode="max")
    print(best_result)




if __name__ == "__main__":
    tune_mnist_asha(num_samples=num_samples)