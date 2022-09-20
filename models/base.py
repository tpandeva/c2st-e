from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tests import c2st_e, c2st, MMDu, mmd2_permutations, me, scf
import numpy as np

class C2ST(pl.LightningModule):
    def forward(self):
        raise NotImplemented()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.long())
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        evals = c2st_e(y=y, logits=y_hat)
        loss = F.cross_entropy(y_hat, y.long())
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == y_hat) / x.shape[0]

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_e', evals, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc, 'val_e': evals}

    def test_le(self, x, y, N_per, alpha=0.05):

        N = x.shape[0]
        f = torch.nn.Softmax()
        x = f(x)
        N1 = y.sum().int()
        STAT = abs((x[y == 1, 0]).type(torch.FloatTensor).mean() - (x[y == 0, 0]).type(torch.FloatTensor).mean())

        STAT_vector = np.zeros(N_per)
        for r in range(N_per):
            ind = np.random.choice(N, N, replace=False)
            ind_X = ind[:N1]
            ind_Y = ind[N1:]
            STAT_vector[r] = abs(
                x[ind_X, 0].type(torch.FloatTensor).mean() - x[ind_Y, 0].type(torch.FloatTensor).mean())
        S_vector = np.sort(STAT_vector)
        threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
        h = 0
        if STAT.item() > threshold:
            h = 1
        return h, threshold, STAT
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        c2ste = c2st_e(y, y_hat)
        c2stp = c2st(y, y_hat)
        hl, thres, c2stl = self.test_le(y_hat, y, 100)
        he = 1 if c2ste>20 else 0
        hp = 1 if c2stp<0.05 else 0
        self.log('test_c2stp', hp, on_epoch=True, prog_bar=True)
        self.log('test_c2stl', hl , on_epoch=True, prog_bar=True)
        self.log('test_c2ste', he, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer


class MMD(pl.LightningModule):
    def forward(self):
        raise NotImplemented()

    def training_step(self, batch, batch_idx):
        x, y = batch
        mmd2, varEst, Kxyxy = self(x, y)
        mmd_value_temp = -1 * mmd2
        mmd_std_temp = torch.sqrt(varEst + 10 ** (-8))
        loss = torch.div(mmd_value_temp, mmd_std_temp)
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        mmd2, varEst, Kxyxy = self(x, y)
        mmd_value_temp = -1 * mmd2
        mmd_std_temp = torch.sqrt(varEst + 10 ** (-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

        self.log('val_loss', STAT_u, on_epoch=True, prog_bar=True)
        return STAT_u

    def test_step(self, batch, batch_idx):
        x, y = batch

        mmd2, varEst, Kxyxy = self(x, y)
        mmd_value_temp = -1 * mmd2
        mmd_std_temp = torch.sqrt(varEst + 10 ** (-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        self.log('test_loss', STAT_u, on_epoch=True, prog_bar=True)

        nx = x.shape[0]
        mmd_value_nn, p_val, rest = mmd2_permutations(Kxyxy, nx, permutations=200)
        self.log('MMD-D P', p_val, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer
