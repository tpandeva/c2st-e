from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tests import c2st_e, c2st, MMDu, mmd2_permutations, me, scf
import numpy as np
class SimpleClassifier(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(hparams.input_size, hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.output_size),
        )
        if hparams.test == "mmd-d":
            self.W = torch.nn.Parameter(torch.randn([hparams.hidden_layer_size, 2]).float(), requires_grad=True)
            self.b = torch.nn.Parameter(torch.randn([1, 2]).float(), requires_grad=True)
        #else:
        #    self.linear2 = nn.Linear(hparams.hidden_layer_size, 2)

        #else:
        #    self.linear1 = nn.Linear(hparams.input_size+1, hparams.hidden_layer_size)
        #    self.linear2 = nn.Linear(hparams.hidden_layer_size, 1)

    def forward(self, x, y=None):
       # if y is not None:
       #     x = torch.concat((x, y.view(-1,1)),1)
        x = self.latent(x)
        #x = F.relu(x)
        if self.hparams.test=="mmd-d":
            x=x.mm(self.W)+self.b

       # else:
        #    x = self.linear2(x)

        if y is None:
            #x = F.relu(x)
            return x
        else:
            x_prob = F.sigmoid(x)
            return x, x_prob

    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch

        if self.hparams.loss == "cross_entropy":
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y.long())
        else:
            _,s = self(x,y)
            _,s0 = self(x, torch.zeros(y.shape[0]))
            _,s1 = self(x, torch.ones(y.shape[0]))
            emp1 = (y[y==1]).sum()/y.shape[0]
            emp0 = 1-emp1
            loss = torch.sum(-torch.log(s)-emp0*(1-s0)-emp1*(1-s1))
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.hparams.loss=="cross_entropy":
            y_hat = self(x)
            y_hat = torch.argmax(y_hat, dim=1)
            acc = torch.sum(y == y_hat)/x.shape[0]
        else:
            _,y_hat = self(x,y)
            y_hat = torch.round(y_hat).int()
            acc = torch.sum(y == y_hat.view(-1,)) / y.shape[0]
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc
    def test_le(self, x,y, N_per, alpha=0.05):

        N = x.shape[0]
        f = torch.nn.Softmax()
        STAT = abs((x[:,0]*y).type(torch.FloatTensor).mean() - (x[:,0]*(1-y)).type(torch.FloatTensor).mean())
        N1 = y.sum().int()
        STAT_vector = np.zeros(N_per)
        for r in range(N_per):
            ind = np.random.choice(N, N, replace=False)
            # divide into new X, Y
            ind_X = ind[:N1]
            ind_Y = ind[N1:]
            # print(indx)
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
        if self.hparams.loss == "cross_entropy":
            y_hat = self(x)
            c2ste = c2st_e(y, y_hat)
            c2stp = c2st(y,y_hat)
            _,thres, c2stl = self.test_le(y_hat, y, 100)
            self.log('test_c2stp', c2stp, on_epoch=True, prog_bar=True)
            self.log('test_c2stl_thres', thres, on_epoch=True, prog_bar=True)
            self.log('test_c2stl_c2stl', c2stl, on_epoch=True, prog_bar=True)
            self.log('test_c2ste', c2ste, on_epoch=True, prog_bar=True)
        else:
            g,_ = self(x,y)
            g1,_ = self(x, torch.ones(y.shape[0]))
            g0,_ = self(x, torch.zeros(y.shape[0]))
            c2ste = c2st_e(y, torch.concat((g, g0, g1), 1), self.hparams.emp1)
            #c2stp = c2st(y, torch.concat((1 - prob, prob), 1))




            self.log('test_c2ste', c2ste, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer



class MMD_DClassifier(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(hparams.input_size, hparams.hidden_layer_size ),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.hidden_layer_size, bias=True),
        )
        self.eps, self.sigma, self.sigma0_u = torch.nn.Parameter(torch.from_numpy(np.random.rand(1) * (10 ** (-10))),  requires_grad=True),\
                                              torch.nn.Parameter(torch.from_numpy(np.sqrt(np.random.rand(1) * 0.3)),  requires_grad=True),\
                                              torch.nn.Parameter(torch.from_numpy(np.sqrt(np.random.rand(1) * 0.002)),  requires_grad=True)


    def forward(self, x,y):
        xy = torch.concat((x,y))
        xy_hat = self.latent(xy)
        mmd2, varEst, Kxyxy = MMDu(xy_hat[0:x.shape[0],:], xy_hat[x.shape[0]:,:], x,y, self.sigma ** 2,self.sigma0_u ** 2, torch.exp(self.eps) / (1 + torch.exp( self.eps)))
        return mmd2, varEst, Kxyxy

    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        """
        x_hat = self(x)
        y_hat = self(y)
        eps = torch.exp(self.eps) / (1 + torch.exp( self.eps))
        sigma = self.sigma ** 2
        sigma0_u = self.sigma0_u ** 2
        """
        mmd2, varEst, Kxyxy = self(x, y)
        mmd_value_temp = -1 * mmd2
        mmd_std_temp = torch.sqrt(varEst + 10 ** (-8))
        loss = torch.div( mmd_value_temp, mmd_std_temp)
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        mmd2, varEst, Kxyxy = self(x, y)
        mmd_value_temp = -1 *mmd2
        mmd_std_temp = torch.sqrt(varEst + 10 ** (-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

        self.log('val_loss',STAT_u, on_epoch=True, prog_bar=True)
        return STAT_u
    def test_step(self, batch, batch_idx):
        x, y = batch

        mmd2, varEst, Kxyxy  = self(x, y)
        mmd_value_temp = -1 * mmd2
        mmd_std_temp = torch.sqrt(varEst + 10 ** (-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        self.log('test_loss', STAT_u, on_epoch=True,prog_bar=True)

        nx = x.shape[0]
        mmd_value_nn, p_val, rest = mmd2_permutations(Kxyxy, nx, permutations=200)
        self.log('MMD-D P', p_val, on_epoch=True,prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer


