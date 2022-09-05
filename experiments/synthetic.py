
from torch.utils.data import Dataset
import numpy as np
from sklearn.utils import check_random_state
import torch
from sklearn.datasets import  make_moons

class Data(Dataset):
    def __init__(self, data, samples, seed, with_labels=True):
        if data.type=="blob":
            sep = 1
            rows = 3
            cols = 3
            """Generate Blob-S for testing type-I error."""
            rs = check_random_state(seed)
            mu = np.zeros(2)
            sigma = np.eye(2)
            X = rs.multivariate_normal(mu, sigma, size=samples)
            Y = rs.multivariate_normal(mu, sigma, size=samples)
            # assign to blobs
            X[:, 0] += rs.randint(rows, size=samples) * sep
            X[:, 1] += rs.randint(cols, size=samples) * sep
            Y[:, 0] += rs.randint(rows, size=samples) * sep
            Y[:, 1] += rs.randint(cols, size=samples) * sep
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y)
        elif data.type=="two_moons":
            dt = [torch.from_numpy(el) for el in
                          make_moons(n_samples=samples, noise=data.noise, random_state=seed)]
            y = dt[1]
            X = dt[0][y==1]
            Y = dt[0][y==0]
        if with_labels:
            self.x = torch.concat((X, Y)).float()
            self.y = torch.concat((torch.ones(int(X.shape[0])), torch.zeros(int(X.shape[0]))))
        else:
            self.x = X.float()
            self.y = Y.float()


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
