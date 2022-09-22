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
from models.nn import (
    Featurizer,
    Discriminator,
    Classifier,
    set_parameters_grad,
    Regressor,
)
from pytorch_lightning import Callback
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset
from sklearn.datasets import make_circles, make_moons
from typing import List
import logging
import numpy as np
import torchvision.models as models
import pickle

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="defaultCIFAR.yaml")
def train(cfg: DictConfig):

    pretrained_model = models.resnet152(pretrained=True)
    model_d = 2048
    pretrained_model.fc = nn.Tanh()
    set_parameters_grad(pretrained_model, requires_grad=False)
    pretrained_model = pretrained_model.cuda()
    pretrained_model.eval()

    is_mmd = False
    data_old = datasets.CIFAR10(
        root=cfg.data.folder,
        download=False,
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(cfg.data.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader_old = torch.utils.data.DataLoader(
        data_old, batch_size=10000, shuffle=True, num_workers=1
    )

    for i, (imgs, Labels) in enumerate(dataloader_old):
        data_old = imgs

    data_new = np.load(cfg.data.file)
    data_new = np.transpose(data_new, [0, 3, 1, 2])
    ind_M = np.random.choice(len(data_new), len(data_new), replace=False)
    data_new = data_new[ind_M]
    TT = transforms.Compose(
        [
            transforms.Resize(cfg.data.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trans = transforms.ToPILImage()
    data_trans = torch.zeros([len(data_new), 3, cfg.data.img_size, cfg.data.img_size])
    data_new = torch.from_numpy(data_new)
    for i in range(len(data_new)):
        d0 = trans(data_new[i])
        data_trans[i] = TT(d0)
    data_new = data_trans
    del data_trans
    resp, rese, resl, resm, resme, resscf = [], [], [], [], [], []
    for s in range(10):
        pl.seed_everything(s)
        ind = np.random.choice(len(data_old), len(data_new), replace=False)
        dataold = data_old[ind, :, :, :]
        if is_mmd:
            # data_s = torch.concat((dataold,data_new))
            # ind_s = np.random.choice(len(data_s), len(data_s), replace=False)
            # data_s = data_s[ind_s]
            data = torch.utils.data.TensorDataset(dataold, data_new)
            train_data, val_data, test_data = torch.utils.data.random_split(
                data, [1000, 0, 1021]
            )
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1021)
        else:
            x = torch.concat((dataold, data_new)).float()
            # x = pretrained_model(x.cuda()).cuda()

            y = torch.concat(
                (torch.ones(int(dataold.shape[0])), torch.zeros(int(data_new.shape[0])))
            )

            data = torch.utils.data.TensorDataset(x, y)
            train_data, val_data, test_data = torch.utils.data.random_split(
                data, [2000, 0, 2042]
            )
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=2042)

        logger.info(OmegaConf.to_yaml(cfg))

        # Initialize the network
        callbacks: List[Callback] = hydra.utils.instantiate(cfg.early_stopping)
        if is_mmd:
            classifier = Featurizer(
                cfg.model
            )  # Regressor(cfg.model, pretrained_model)#MMD_DClassifier(cfg.model)
        else:
            classifier = Classifier(
                cfg.model, pretrained_model
            )  # Featurizer(cfg.model)#MMD_DClassifier(cfg.model)

        # Train

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=cfg.data.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=cfg.data.batch_size
        )
        trainer = pl.Trainer(**cfg.trainer, accelerator="gpu", devices=1)
        trainer.fit(classifier, train_loader)

        stats = trainer.test(classifier, test_loader)
        # Test
        if is_mmd:
            resp.append(stats[0]["MMD-D P"])

        else:
            resp.append(stats[0]["test_c2stp"])
            rese.append(stats[0]["test_c2ste"])
            resl.append(stats[0]["test_c2stl"])

        # resm.append(stats[0]['MMD-D P'])
        del classifier, trainer
        print(resp)
        print(rese)
        print(resl)
        res = [resp, rese, resl]
        open_file = open(f"mmd.pickle", "wb")
        pickle.dump(res, open_file)
        open_file.close()


if __name__ == "__main__":
    train()
