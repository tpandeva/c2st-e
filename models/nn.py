from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tests import c2st_e, c2st

class SimpleClassifier(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.linear1 = nn.Linear(hparams.input_size, hparams.hidden_layer_size)
        self.linear2 = nn.Linear(hparams.hidden_layer_size, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x

    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == y_hat)/x.shape[0]
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        c2ste = c2st_e(y, y_hat)
        c2stp = c2st(y, y_hat)

        self.log('test_c2stp', c2stp, on_epoch=True, prog_bar=True)

        self.log('test_c2ste', c2ste, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer





