import numpy as np
import torch
from torch.distributions import Normal


def c2st(y, y_hat):
    # H0: accuracy=0.5 vs H1: accuracy>0.5
    y_hat = torch.argmax(y_hat, dim=1)
    accuracy = torch.sum(y == y_hat) / y.shape[0]
    n_te = y.shape[0]
    stat = 2 * np.sqrt(n_te) * (accuracy - 0.5)
    pval = 1 - Normal(0, 1).cdf(stat)
    return pval


def c2st_l(S, y, N_per, alpha, model_C2ST):
    """run C2ST-L."""
    N = S.shape[0]
    N1 = int(torch.sum(y).to_numpy())
    f = torch.nn.Softmax()
    output = f(model_C2ST(S))
    STAT = abs((y * output.type(torch.FloatTensor)).mean() - ((1 - y) * output.type(torch.FloatTensor)).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(
            output[ind_X, 0].type(torch.FloatTensor).mean() - output[ind_Y, 0].type(torch.FloatTensor).mean()
        )
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT
