import numpy as np
import torch
from torch.distributions import Normal
from sklearn.metrics import accuracy_score


def c2st(y,y_hat):
    # H0: accuracy=0.5 vs H1: accuracy>0.5
    y_hat = torch.argmax(y_hat, dim=1)
    accuracy = torch.sum(y == y_hat) / y.shape[0]
    n_te = y.shape[0]
    stat = 2 * np.sqrt(n_te) * (accuracy - 0.5)
    pval = 1 - Normal(0, 1).cdf(torch.tensor(stat))
    return pval



