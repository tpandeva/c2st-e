import pytorch_lightning as pl
from models.nn import SimpleClassifier
from pytorch_lightning import Callback
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset
from sklearn.datasets import make_circles, make_moons
from typing import List
import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path='configs', config_name='default.yaml')
def train(cfg: DictConfig):
    # The decorator is enough to let Hydra load the configuration file.

    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))

    # We recover the original path of the dataset:

    # Load data
    train_data = [torch.from_numpy(el) for el in make_moons(n_samples=cfg.data.train_samples, noise=1)]
    val_data = [torch.from_numpy(el) for el in make_moons(n_samples=cfg.data.val_samples, noise=1)]
    test_data =[torch.from_numpy(el) for el in make_moons(n_samples=cfg.data.test_samples, noise=1)]

    # Wrap data with appropriate data loaders
    train_loader = torch.utils.data.DataLoader(TensorDataset(train_data[0].float(),train_data[1]), batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(TensorDataset(val_data[0].float(),val_data[1]), batch_size=cfg.data.batch_size)
    test_loader = torch.utils.data.DataLoader(TensorDataset(test_data[0].float(),test_data[1]), batch_size=cfg.data.batch_size)

    pl.seed_everything(cfg.seed)

    # Initialize the network
    callbacks: List[Callback] = hydra.utils.instantiate(cfg.early_stopping)
    classifier = SimpleClassifier(cfg.model)
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)
    trainer.fit(classifier, train_loader, val_loader)
    trainer.test(classifier,test_loader)

if __name__ == "__main__":
    train()