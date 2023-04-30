import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import numpy as np

class MNIST_MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, sync_dist=True)

        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        acc = np.mean((torch.argmax(y_pred, dim=1) == y).cpu().numpy())
        self.log("val_acc", acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        acc = np.mean((torch.argmax(y_pred, dim=1) == y).cpu().numpy())
        self.log("test_acc", acc, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=2)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=2)

if __name__ == "__main__":
    cli = LightningCLI(MNIST_MLP, MNISTDataModule, seed_everything_default=0)

