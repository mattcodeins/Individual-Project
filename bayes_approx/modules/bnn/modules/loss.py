# import warnings

from torch.nn import Module
# from torch.nn import functional as F
# from torch.nn import _reduction as _Reduction

import modules.bnn.functional as BF


class _Loss(Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction


class GaussianKLLoss(_Loss):
    """
    Loss for calculating KL divergence of baysian neural network model.

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
    """
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, model):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        return BF.gaussian_kl_loss(model)


class nELBO(_Loss):
    def __init__(self, nll_loss, kl_loss):
        super(nELBO, self).__init__()
        self.nll_loss = nll_loss
        self.kl_loss = kl_loss

    def forward(self, model, loss_args, minibatch_ratio):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        return BF.nelbo(model, loss_args, minibatch_ratio, nll_loss=self.nll_loss, kl_loss=self.kl_loss)
