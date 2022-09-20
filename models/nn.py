from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tests import c2st_e, c2st, MMDu, mmd2_permutations, me, scf
import numpy as np
from .base import C2ST, MMD
from .utils import set_parameters_grad


class SimpleClassifier(C2ST):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(hparams.input_size, hparams.hidden_layer_size),
            torch.nn.BatchNorm1d(hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.hidden_layer_size),
            torch.nn.BatchNorm1d(hparams.hidden_layer_size),
           # torch.nn.ReLU(),
           # torch.nn.Linear(hparams.hidden_layer_size, hparams.hidden_layer_size),
           # torch.nn.BatchNorm1d(hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size,hparams.output_size),#

        )
        if hparams.test == "mmd-d":
           self.W = torch.nn.Parameter(torch.randn([hparams.output_size, 2]).float(), requires_grad=True)
           self.b = torch.nn.Parameter(torch.randn([1, 2]).float(), requires_grad=True)

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



class MMD_DClassifier(MMD):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(hparams.input_size, hparams.hidden_layer_size),
            torch.nn.BatchNorm1d(hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.hidden_layer_size),
            torch.nn.BatchNorm1d(hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.hidden_layer_size),
            torch.nn.BatchNorm1d(hparams.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams.hidden_layer_size, hparams.output_size),
        )
        self.eps, self.sigma, self.sigma0_u = torch.nn.Parameter(torch.from_numpy(np.random.rand(1) * (10 ** (-10))),  requires_grad=True),\
                                              torch.nn.Parameter(torch.from_numpy(np.sqrt(np.random.rand(1) * 0.3)),  requires_grad=True),\
                                              torch.nn.Parameter(torch.from_numpy(np.sqrt(np.random.rand(1) * 0.002)),  requires_grad=True)


    def forward(self, x,y):
        xy = torch.concat((x,y))
        xy_hat = self.latent(xy)
        mmd2, varEst, Kxyxy = MMDu(xy_hat[0:x.shape[0],:], xy_hat[x.shape[0]:,:], x,y, self.sigma ** 2,self.sigma0_u ** 2, torch.exp(self.eps) / (1 + torch.exp( self.eps)))
        return mmd2, varEst, Kxyxy


class Discriminator(C2ST):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.0)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters,0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(hparams.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = hparams.output_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Flatten(),
            nn.Linear(128 * ds_size ** 2, 100),         
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax())

    def forward(self, img):
        out = self.model(img)
        #out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class Featurizer(MMD):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1,bias=False),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(hparams.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = hparams.output_size// 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 300))
        self.eps, self.sigma, self.sigma0_u = torch.nn.Parameter(torch.log(torch.from_numpy(np.random.rand(1) * (10 ** (-10)))),
                                                                 requires_grad=True), \
                                              torch.nn.Parameter(torch.from_numpy(np.ones(1) * np.sqrt(2 * 32 * 32)),
                                                                 requires_grad=True), \
                                              torch.nn.Parameter(torch.from_numpy(np.sqrt(np.ones(1) * 0.005)),
                                                                 requires_grad=True)

    def forward(self, x,y):
        img = torch.concat((x,y))
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)
        si = self.sigma ** 2
        si0 = self.sigma0_u ** 2
        eps = torch.exp(self.eps) / (1 + torch.exp(self.eps))
        mmd2, varEst, Kxyxy = MMDu(feature[0:x.shape[0], :], feature[x.shape[0]:, :], x.view(x.shape[0],-1), y.view(y.shape[0],-1), si,
                                   si0, eps)
        return mmd2, varEst, Kxyxy

    
    
class Classifier(C2ST):
    def __init__(self, hparams: DictConfig, pretrained_model):
        super().__init__()
        self.save_hyperparameters(hparams)     
        model_d = 2048
       
        self.pm = pretrained_model
        self.model = nn.Sequential(nn.Linear(model_d, 2),nn.Softmax())
       
    def forward(self, img):
        feat = self.pm(img)
        out = self.model(feat)
        return out
    
class Regressor(pl.LightningModule):
    def __init__(self, hparams: DictConfig, pretrained_model):
        super().__init__()
        self.save_hyperparameters(hparams)     
        model_d = 2048
       
        self.pm = pretrained_model
     
        self.model = nn.Sequential(nn.Linear(model_d, 300),
                                   nn.ReLU(),
                                  nn.Linear(300, 300))
        self.eps, self.sigma, self.sigma0_u = torch.nn.Parameter(torch.from_numpy(np.random.rand(1) * (10 ** (-10))),  requires_grad=True),\
                                              torch.nn.Parameter(torch.from_numpy(np.sqrt(np.random.rand(1) * 0.3)),  requires_grad=True),\
                                              torch.nn.Parameter(torch.from_numpy(np.sqrt(np.random.rand(1) * 0.002)),  requires_grad=True)


     
    def forward(self, x,y):
        img = torch.concat((x,y))
        out = self.pm(img)
        out = self.model(out)
        si = self.sigma ** 2
        si0 = self.sigma0_u ** 2
        eps = torch.exp(self.eps) / (1 + torch.exp(self.eps))
        mmd2, varEst, Kxyxy = MMDu(out[0:x.shape[0], :], out[x.shape[0]:, :], x.view(x.shape[0],-1), y.view(y.shape[0],-1), si,
                                   si0, eps)
        return mmd2, varEst, Kxyxy

        
