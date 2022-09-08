import pytorch_lightning as pl
from models.nn import SimpleClassifier, MMD_DClassifier
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
from sklearn.utils import check_random_state
from torch.distributions.normal import Normal
from tests import me, scf
from experiments.synthetic import Data
logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split
@hydra.main(config_path='configs', config_name='default.yaml')
def train(cfg: DictConfig):
    powerp, powere, powerl, powerm, powerme, powerscf = np.zeros((10,100)), np.zeros((10,100)), np.zeros((10,100)),\
                                                    np.zeros((10,100)), np.zeros((10,100)), np.zeros((10,100))
    l=0
    if cfg.data.type == "blob" or cfg.data.type == "blob-2":
        n_list = list(range(90,990,90))
    else:
        n_list = list(range(100,1100,100))
    for samples in [270]:#n_list:

        for s in range(1):
            logger.info(OmegaConf.to_yaml(cfg))

            if False:
                x = torch.concat((Normal(0,1).sample(sample_shape=[3*samples]).view(-1,1),
                                  Normal(0,1).sample(sample_shape=[3*samples]).view(-1,1)),1)
                y = torch.concat((torch.zeros(int(1.5*samples)),torch.ones(int(1.5*samples))))
                dataset = TensorDataset(x.float(),y)
                train_data, val_data, test_data = torch.utils.data.random_split(dataset, [samples, samples,samples])
            # Wrap data with appropriate data loaders
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
                ind = train_data.indices
                emp1 = torch.mean(y[ind])
                val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size)
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=samples)
                overrides = [f"model.emp1={emp1}"]

                cfg = hydra.compose("default.yaml", overrides=overrides)


            pl.seed_everything(cfg.seed)

            # Initialize the network
            callbacks: List[Callback] = hydra.utils.instantiate(cfg.early_stopping)
            classifier = SimpleClassifier(cfg.model)#MMD_DClassifier(cfg.model)

            # Train

            train_data = Data(cfg.data, samples, s)
            val_data = Data(cfg.data, samples, s+12345)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=samples, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=samples)
            trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)
            trainer.fit(classifier, train_loader, val_loader)

            train_data = Data(cfg.data, samples, s, with_labels=False)
            val_data = Data(cfg.data, samples, s+12345, with_labels=False)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size)

            # Initialize the network
            callbacks: List[Callback] = hydra.utils.instantiate(cfg.early_stopping)
            classifierMMD = MMD_DClassifier(cfg.model)  # MMD_DClassifier(cfg.model)

            # Train
            trainerMMD = pl.Trainer(**cfg.trainer, callbacks=callbacks)
            trainerMMD.fit(classifierMMD, train_loader, val_loader)

            # Train me and scf
            test_locs_ME, gwidth_ME = me(train_data[:][0], train_data[:][1], alpha=0.05, is_train=True, test_locs=1, gwidth=1,
                                   J=5, seed=15)

            test_freqs_SCF, gwidth_SCF = scf(train_data[:][0], train_data[:][1], alpha=0.05, is_train=True, test_freqs=1,
                                       gwidth=1, J=5, seed=15)

            # Test
            resp, rese, resl, resm, resme, resscf = [], [], [], [], [], []
            for k in range(100):

                dataset  = Data(cfg.data, samples, 1000*(k+1)+s, with_labels=True)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size = 2*samples)

                stats = trainer.test(classifier, test_loader)
                resp.append(stats[0]['test_c2stp'])
                rese.append(stats[0]['test_c2ste'])
                resl.append(stats[0]['test_c2stl_c2stl'] > stats[0]['test_c2stl_thres'])
                dataset  = Data(cfg.data, samples, 1000*(k+1)+s, with_labels=False)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size= samples)
                stats = trainerMMD.test(classifierMMD, test_loader)
                resm.append(stats[0]['MMD-D P'])
                h_ME = me(dataset[:][0], dataset[:][1], alpha=0.05, is_train=False, test_locs=test_locs_ME,
                          gwidth=gwidth_ME, J=5, seed=15)
                resme.append(h_ME)
                h_SCF = scf(dataset[:][0], dataset[:][1], alpha=0.05, is_train=False, test_freqs=test_freqs_SCF,
                            gwidth=gwidth_SCF, J=5, seed=15)
                resscf.append(h_SCF)
            del classifier, trainer, classifierMMD, trainerMMD
            resp = np.array(resp)
            resm = np.array(resm)
            rese = np.array(rese)
            resl = np.array(resl)
            powerp[l,s] = np.mean(resp<0.05)
            powerm[l,s] = np.mean(resm < 0.05)
            powerl[l,s] = np.mean(resl)
            powerme[l,s] = np.mean(resme)
            powerscf[l,s] = np.mean(resscf)
            powere[l,s] = np.mean(rese > 20)

            print("Type 1 E", powere)
            print("Type 1 P", powerp)
            print("Type 1 L", powerl)
            print("Type 1 M", powerm)
            print("Type 1 me", powerme)
            print("Type 1 scf", powerscf)
        l += 1
    open_file = open("powerBlob.pickle", "wb")

    pickle.dump([powerp, powere, powerl, powerm, powerme, powerscf], open_file)

    open_file.close()
    return powerp, powere, powerl, powerm, powerme, powerscf

if __name__ == "__main__":
    train()