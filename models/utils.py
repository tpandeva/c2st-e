


def set_parameters_grad(model, requires_grad):
    '''update requires_grad for all paramters in model'''
    for param in model.parameters():
        param.requires_grad = requires_grad