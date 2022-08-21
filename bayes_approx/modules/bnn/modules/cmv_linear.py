import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CMVBayesLinear(nn.Module):
    """
    Learnable single prior std shared by all weights and biases.
    Prior mean is zero for all weights and biases.
    """
    def __init__(self, in_features, out_features,
                 _hyperprior_alpha_param, _hyperprior_beta_param, _hyperprior_delta_param,
                 bias=True, init_std=0.05, sqrt_width_scaling=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype, 'requires_grad': True}
        super(CMVBayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # approximate posterior parameters (Gaussian)
        self.weight_mean = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self._weight_std_param = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.bias = bias
        if self.bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self._bias_std_param = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('_bias_std_param', None)
        self.reset_parameters(init_std)

        # hyperprior parameters
        self._hyperprior_alpha_param = _hyperprior_alpha_param
        self._hyperprior_beta_param = _hyperprior_beta_param
        self._hyperprior_delta_param = _hyperprior_delta_param

        prior_mean = 0
        self.register_buffer('prior_weight_mean_expect', torch.full_like(self.weight_mean, prior_mean))
        self.prior_weight_std_expect = np.sqrt(_hyperprior_beta_param/_hyperprior_alpha_param)
        if self.bias:
            self.register_buffer('prior_bias_mean_expect', torch.full_like(self.bias_mean, prior_mean))
            self.prior_bias_std_expect = _hyperprior_beta_param/_hyperprior_alpha_param
        else:
            self.register_buffer('prior_bias_mean', None)
            self.register_buffer('prior_bias_std', None)

    # def extra_repr(self):
    #     repr = "in_features={}, out_features={}, bias={}".format(
    #         self.in_features, self.out_features, self.bias is not None
    #     )
    #     weight_std = self.prior_weight_std.data.flatten()[0]
    #     if torch.allclose(weight_std, self.prior_weight_std):
    #         repr += f", weight prior std={weight_std.item():.2f}"
    #     bias_std = self.prior_bias_std.flatten()[0]
    #     if torch.allclose(bias_std, self.prior_bias_std):
    #         repr += f", bias prior std={bias_std.item():.2f}"
    #     return repr

    def reset_parameters(self, init_std=0.05):
        # nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        bound = 1. / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mean, -bound, bound)
        nn.init.constant_(self._weight_std_param, np.log(np.exp(init_std) - 1))
        if self.bias:
            nn.init.uniform_(self.bias_mean, -bound, bound)
            nn.init.constant_(self._bias_std_param, np.log(np.exp(init_std) - 1))

    # define the q distribution standard deviations with property decorator
    @property
    def weight_std(self):
        return torch.log(1 + torch.exp(self._weight_std_param))

    @property
    def bias_std(self):
        return torch.log(1 + torch.exp(self._bias_std_param))

    @property
    def hyperprior_alpha(self):
        return torch.log(1 + torch.exp(self._hyperprior_alpha_param))

    @property
    def hyperprior_beta(self):
        return torch.log(1 + torch.exp(self._hyperprior_beta_param))

    @property
    def hyperprior_delta(self):
        return torch.log(1 + torch.exp(self._hyperprior_delta_param))

    # forward pass using reparam trick
    def forward(self, input):
        weight = self.weight_mean + self.weight_std * torch.randn_like(self.weight_std)
        if self.bias:
            bias = self.bias_mean + self.bias_std * torch.randn_like(self.bias_std)
        else:
            bias = None
        return F.linear(input, weight, bias)


# construct a BNN with learnable prior (std)
def make_linear_cmv_bnn(layer_sizes, alpha, beta, delta, activation='ReLU', **layer_kwargs):
    nonlinearity = getattr(nn, activation)() if isinstance(activation, str) else activation
    bnn = nn.Sequential()

    bnn.register_buffer(
        '_hyperprior_alpha_param',
        torch.tensor(np.log(np.exp(alpha) - 1))
    )
    bnn.register_buffer(
        '_hyperprior_beta_param',
        torch.tensor(np.log(np.exp(beta) - 1))
    )
    bnn.register_buffer(
        '_hyperprior_delta_param',
        torch.tensor(np.log(np.exp(delta) - 1))
    )

    for i, (dim_in, dim_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        bnn.add_module(f'CMVBayesLinear{i}', CMVBayesLinear(
            dim_in, dim_out,
            bnn._hyperprior_alpha_param,
            bnn._hyperprior_beta_param,
            bnn._hyperprior_delta_param,
            **layer_kwargs
        ))
        if i < len(layer_sizes) - 2:
            bnn.add_module(f'Nonlinearity{i}', nonlinearity)
    return bnn
