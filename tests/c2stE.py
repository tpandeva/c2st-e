
import torch
def c2st_e(y, logits, num_batches=None):

    # H0
    # p(y|x) of MLE under H0: p(y|x) = p(y), is just the empirical frequency of y in the test data.
    emp_freq_class0 = 1-(y[y==1]).sum()/y.shape[0]
    emp_freq_class1 = (y[y==1]).sum()/y.shape[0]

    # H1
    # Probabilities under empirical model (using train data)
    prob = torch.exp(logits)
    prob = prob/prob.sum(1).view(-1,1)
    pred_prob_class0 = prob[:, 0]
    pred_prob_class1 = prob[:, 1]

    if num_batches is None:
        log_eval = torch.sum(y * torch.log(pred_prob_class1 / emp_freq_class1) + (1 - y) * torch.log(
            pred_prob_class0 / emp_freq_class0)).double()
        eval = torch.exp(log_eval)

    else:
        eval = 1
        ratios = y * pred_prob_class1 / emp_freq_class1 + (1 - y) * pred_prob_class0 / emp_freq_class0
        ind = torch.randperm(ratios.shape[0])
        ratios = ratios[ind]
        ratio_batches = [ratios[i::num_batches] for i in range(num_batches)]
        for i in range(num_batches):
            eval = eval * torch.mean(ratio_batches[i])

    # E-value
    return eval