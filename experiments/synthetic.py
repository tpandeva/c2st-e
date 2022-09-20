
from torch.utils.data import Dataset
import numpy as np
from sklearn.utils import check_random_state
import torch
from sklearn.datasets import  make_moons
def sample_blobs_Q(N1, sigma_mx_2, rows=3, cols=3, rs=None):
    """Generate Blob-D for testing type-II error (or test power)."""
    rs = check_random_state(rs)
    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    X = rs.multivariate_normal(mu, sigma, size=N1)
    Y = rs.multivariate_normal(mu, np.eye(2), size=N1)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=N1)
    X[:, 1] += rs.randint(cols, size=N1)
    Y_row = rs.randint(rows, size=N1)
    Y_col = rs.randint(cols, size=N1)
    locs = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        Y = np.where(ind2, np.matmul(Y,L) + locs[i], Y)
    return X, Y

class Data(Dataset):
    def __init__(self, data, samples, seed, with_labels=True):
        if data.type=="blob":
            sep = data.sep
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
        elif data.type=="blob-2":
            sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
            sigma_mx_2 = np.zeros([9, 2, 2])
            for i in range(9):
                sigma_mx_2[i] = sigma_mx_2_standard
                if i < 4:
                    sigma_mx_2[i][0, 1] = -0.02 - 0.002 * i
                    sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
                if i == 4:
                    sigma_mx_2[i][0, 1] = 0.00
                    sigma_mx_2[i][1, 0] = 0.00
                if i > 4:
                    sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i - 5)
                    sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i - 5)
            X,Y=sample_blobs_Q(samples, sigma_mx_2, rs=seed)
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y)
        elif data.type=="two_moons":
            dt = [torch.from_numpy(el) for el in
                          make_moons(n_samples=samples, noise=data.noise, random_state=seed)]
            y = dt[1]
            X = dt[0][y==1]
            Y = dt[0][y==0]

        if with_labels:
            ind = np.random.choice(len(Y)+len(X), len(Y)+len(X), replace=False)
            self.x = torch.concat((X, Y)).float()
            self.x = self.x[ind]
            self.y = torch.concat((torch.ones(int(X.shape[0])),
                                   torch.zeros(int(X.shape[0]))))
            self.y = self.y[ind]
        else:
            self.x = X.float()
            self.y = Y.float()


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
