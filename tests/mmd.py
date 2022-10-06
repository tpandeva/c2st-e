import numpy as np
import torch


def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist < 0] = 0
    return Pdist


def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx, Kxy), 1)
    Kyxy = torch.cat((Kxy.transpose(0, 1), Ky), 1)
    Kxyxy = torch.cat((Kxxy, Kyxy), 0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None

    hh = Kx + Ky - Kxy - Kxy.transpose(0, 1)
    V1 = torch.dot(hh.sum(1) / ny, hh.sum(1) / ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4 * (V1 - V2 ** 2)
    if varEst == 0.0:
        print("error!!" + str(V1))
    return mmd2, varEst, Kxyxy


def MMDu(
    X,
    Y,
    X_org,
    Y_org,
    sigma,
    sigma0=0.1,
    epsilon=10 ** (-10),
    is_smooth=True,
    is_var_computed=True,
    use_1sample_U=True,
):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    L = 1  # generalized Gaussian (if L>1)
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_smooth:
        Kx = (1 - epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma)) ** L + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1 - epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma)) ** L + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1 - epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma)) ** L + epsilon * torch.exp(
            -Dxy_org / sigma
        )
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)


def mmd2_permutations(K, n_X, permutations=500):
    """
    Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero(as_tuple=False)[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero(as_tuple=False)[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    return est.item(), p_val.item(), rest


def mmd_d(
    Fea,
    N_per,
    N1,
    Fea_org,
    sigma,
    sigma0,
    alpha,
    device,
    dtype,
    epsilon=10 ** (-10),
    is_smooth=True,
):
    """run two-sample test (TST) using deep kernel kernel."""
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, epsilon, is_smooth)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    mmd_value_nn, p_val, rest = mmd2_permutations(Kxyxy, nx, permutations=200)
    if p_val > alpha:
        h = 0
    else:
        h = 1
    threshold = "NaN"
    return h, threshold, mmd_value_nn
