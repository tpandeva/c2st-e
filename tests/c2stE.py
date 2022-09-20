import numpy as np
import torch
def c2st_e_permutation(y, logits, N_per=100,alpha=0.05):
    """run C2ST-L."""
    STAT = c2st_e(y, logits)
    N =logits.shape[0]
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # print(indx)
        STAT_vector[r] = c2st_e(y[ind], logits[ind])
    S_vector = np.sort(-STAT_vector)
    threshold = -S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() < threshold:
        h = 1
    return h, STAT
def c2st_e(y, logits, emp1 = None, num_batches=None):

    # H0
    # p(y|x) of MLE under H0: p(y|x) = p(y), is just the empirical frequency of y in the test data.
    emp_freq_class0 = 1-(y[y==1]).sum()/y.shape[0]
    emp_freq_class1 = (y[y==1]).sum()/y.shape[0]

    # H1
    # Probabilities under empirical model (using train data)
    #if logits.shape[1]==2:
    f = torch.nn.Softmax()
    prob =f(logits)
    #prob = prob/prob.sum(1).view(-1,1)

    pred_prob_class0 = prob[:, 0]
    pred_prob_class1 = prob[:, 1]

    if num_batches is None:
        log_eval = torch.sum(y * torch.log(pred_prob_class1 / emp_freq_class1) + (1 - y) * torch.log(
        pred_prob_class0 / emp_freq_class0)).double()
        eval = torch.exp(log_eval)

    else:
        eval = 0
        ratios = y *  torch.log(pred_prob_class1 / emp_freq_class1) + (1 - y) *  torch.log(pred_prob_class0 / emp_freq_class0)
        ind = torch.randperm(ratios.shape[0])
        ratios = ratios[ind]
        ratio_batches = [ratios[i::num_batches] for i in range(num_batches)]
        for i in range(num_batches):
            eval = eval + torch.exp(torch.sum(ratio_batches[i]))
        eval = eval/num_batches
    """else:
        g = logits[:, 0]
        g0 = logits[:, 1]
        g1 = logits[:, 2]
        py_x = torch.exp(g)*(y*emp1 +(1-y)*(1-emp1))/(torch.exp(g0)*(1-emp1)+emp1*torch.exp(g1))
        ratios = y * py_x / emp_freq_class1 + (1 - y) * py_x / emp_freq_class0
        eval = torch.exp(torch.log(torch.sum(ratios)))
    """
    # E-value
    return eval