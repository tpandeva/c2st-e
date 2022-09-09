import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import pytorch_lightning as pl
from models.nn import Featurizer
from pytorch_lightning import Callback
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset
from sklearn.datasets import make_circles, make_moons
from typing import List
import logging
import numpy as np
import pickle
from tests import me, scf
from experiments.synthetic import Data
logger = logging.getLogger(__name__)
@hydra.main(config_path='configs', config_name='defaultCIFAR.yaml')
def train(cfg: DictConfig):
    data_old = datasets.CIFAR10(root=cfg.data.folder, download=False, train=False,
                                    transform=transforms.Compose([
                                        transforms.Resize(cfg.data.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
    dataloader_old = torch.utils.data.DataLoader(data_old, batch_size=10000,
                                                  shuffle=True, num_workers=1)

    for i, (imgs, Labels) in enumerate(dataloader_old):
        data_old = imgs


    data_new =np.load(cfg.data.file)
    data_new = np.transpose(data_new, [0, 3, 1, 2])
    ind_M = np.random.choice(len(data_new), len(data_new), replace=False)
    data_new = data_new[ind_M]
    TT = transforms.Compose([transforms.Resize(cfg.data.img_size), transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans = transforms.ToPILImage()
    data_trans = torch.zeros([len(data_new), 3,cfg.data.img_size, cfg.data.img_size])
    data_new = torch.from_numpy(data_new)
    for i in range(len(data_new)):
        d0 = trans(data_new[i])
        data_trans[i] = TT(d0)
    data_new = data_trans
    del data_trans

    samples = 1000
    ind = np.random.choice(len(data_old), len(data_new),replace=False)
    data_old = data_old[ind, :,:,:]
    data = (data_old,data_new)
    train_data, val_data, test_data =torch.utils.data.random_split(data, [1000, 200, 821])
    logger.info(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

            # Initialize the network
    callbacks: List[Callback] = hydra.utils.instantiate(cfg.early_stopping)
    classifier = Featurizer(cfg.model)#MMD_DClassifier(cfg.model)

            # Train

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size)
    trainer = pl.Trainer(**cfg.trainer, devices=0, callbacks=callbacks)
    trainer.fit(classifier, train_loader, val_loader)


            # Initialize the network
    resp, rese, resl, resm, resme, resscf = [], [], [], [], [], []


    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 2*samples)

    stats = trainer.test(classifier, test_loader)
    resm.append(stats[0]['MMD-D P'])
    del classifier, trainer




if __name__ == "__main__":
    train()