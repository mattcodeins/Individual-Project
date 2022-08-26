import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLGBayesLinear(nn.Module):
    """Applies a linear transformation to the incoming data: y = xW^T + b, where
       the weight W and bias b are sampled from the approximate q distribSution.
    """
    def __init__(self, in_features, out_features, prior_weight_std=1.0, prior_bias_std=1.0,
                 bias=True, init_std=0.05, sqrt_width_scaling=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype, 'requires_grad': True}
        super(MLGBayesLinear, self).__init__()
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

        # prior parameters (Gaussian)
        prior_mean = 0.0
        if sqrt_width_scaling:  # prior variance scales as 1/in_features
            prior_weight_std /= self.in_features**0.5
        # prior parameters are registered as constants
        self.register_buffer('prior_weight_mean', prior_mean * torch.ones(out_features, in_features))
        self.register_buffer('prior_weight_std', prior_weight_std * torch.ones(out_features, in_features))
        if self.bias:
            self.register_buffer('prior_bias_mean', prior_mean * torch.ones(out_features))
            self.register_buffer('prior_bias_std', prior_bias_std * torch.ones(out_features))
        else:
            self.register_buffer('prior_bias_mean', None)
            self.register_buffer('prior_bias_std', None)

    def extra_repr(self):
        repr = "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
        weight_std = self.prior_weight_std.data.flatten()[0]
        if torch.allclose(weight_std, self.prior_weight_std):
            repr += f", weight prior std={weight_std.item():.2f}"
        bias_std = self.prior_bias_std.flatten()[0]
        if torch.allclose(bias_std, self.prior_bias_std):
            repr += f", bias prior std={bias_std.item():.2f}"
        return repr

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

    # forward pass using reparam trick
    def forward(self, input):
        weight_eps = self.weight_mean + self.weight_std * torch.randn_like(self.weight_std)
        weight = self.prior_weight_mean + self.prior_weight_std * weight_eps
        if self.bias:
            bias_eps = self.bias_mean + self.bias_std * torch.randn_like(self.bias_std)
            bias = self.prior_bias_mean + self.prior_bias_std * bias_eps
        else:
            bias_eps = None; bias = None
        return F.linear(input, weight, bias)


# construct a BNN
def make_mlg_linear_bnn(layer_sizes, activation='LeakyReLU', **layer_kwargs):
    nonlinearity = getattr(nn, activation)() if isinstance(activation, str) else activation
    net = nn.Sequential()
    for i, (dim_in, dim_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        net.add_module(f'MLGBayesLinear{i}', MLGBayesLinear(dim_in, dim_out, **layer_kwargs))
        if i < len(layer_sizes) - 2:
            net.add_module(f'Nonlinearity{i}', nonlinearity)
    return net
