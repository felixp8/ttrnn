import numpy as np
import torch
import torch.nn as nn

from .base import RNNBase, RNNCellBase
from .weights import RNNWeights


class rateRNNCell(RNNCellBase):
    """Discretized Rate RNN

    h_t = (1 - alpha) * h_{t-1} + alpha * (W_rec @ r_{t-1} + W_in @ u_t + b_x)
    r_t = f(h_t)
    o_t = g(W_out @ r_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'nonlinearity', 'bias']

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='relu', 
                 dt=10, tau=50, init_config={}, noise_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(rateRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        self.weights = RNNWeights(input_size=input_size, hidden_size=hidden_size, bias=bias, init_config=init_config, **factory_kwargs)

        self.dt = dt
        self.tau = tau

        if self.nonlinearity == 'relu': # for consistency with Torch
            self.hfn = nn.ReLU()
        elif self.nonlinearity == 'tanh': # for consistency with Torch
            self.hfn = nn.Tanh()
        elif hasattr(nn, self.nonlinearity):
            self.hfn = getattr(nn, self.nonlinearity)
        else:
            raise ValueError
    
        self.use_noise = noise_kwargs.get('use_noise', False)
        if self.use_noise:
            self.noise_type = noise_kwargs.get('noise_type', 'normal')
            self.noise_params = noise_kwargs.get('noise_params', {})

        self.reset_parameters()
    
    @property
    def weight_ih(self):
        return self.weights.get_weight_ih(cached=True)
    
    @property
    def weight_hh(self):
        return self.weights.get_weight_hh(cached=True)
    
    @property
    def bias(self):
        return self.weights.get_bias(cached=True)
    
    @property
    def alpha(self):
        # TODO: add support for learnable taus for each dim
        # and maybe learnable dts?
        return self.dt / self.tau
    
    def reset_parameters(self):
        self.weights.reset_parameters()
        self.weights(cached=False)
    
    def set_noise(self, use_noise=None, noise_kwargs={}):
        # need to allow for tau-dependent noise
        if use_noise:
            self.use_noise = use_noise
        if noise_kwargs:
            self.noise_type = noise_kwargs.get('noise_type', self.noise_type)
            self.noise_params = noise_kwargs.get('noise_params', self.noise_params)
 
    def noise(self, device=None, dtype=None):
        if not self.use_noise:
            return 0. # torch.zeros((1, self.hidden_size), **self.factory_kwargs)
        else:
            out = torch.empty((1, self.hidden_size), device=device, dtype=dtype)
            return getattr(torch, self.noise_type)(size=(1, self.hidden_size), out=out, **self.noise_params)
    
    def forward(self, input, hx):
        weights = self.weights(cached=True)
        weight_ih = weights['weight_ih']
        weight_hh = weights['weight_hh']
        bias = weights['bias']
        device, dtype = input.device, input.dtype
        if bias is None:
            hx = hx * (1 - self.alpha) + self.alpha * (
                torch.mm(input, weight_ih.t()) + 
                torch.mm(self.hfn(hx), weight_hh.t()) + 
                self.noise(device, dtype))
        else:
            hx = hx * (1 - self.alpha) + self.alpha * (
                torch.mm(input, weight_ih.t()) + 
                torch.mm(self.hfn(hx), weight_hh.t()) + 
                bias + self.noise(device, dtype))
        return hx


class rateRNN(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size, bias=True, nonlinearity='relu', 
                 dt=10, tau=50, learnable_h0=True, batch_first=False,
                 init_config={}, noise_kwargs={}, output_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = rateRNNCell(input_size, hidden_size, bias, nonlinearity, dt, tau, init_config, noise_kwargs, device, dtype)
        # readout = self.configure_output(rnn_cell=rnn_cell, hidden_size=hidden_size, output_size=output_size, **output_kwargs)
        super(rateRNN, self).__init__(rnn_cell, input_size, hidden_size, output_size, batch_first, output_kwargs)
        self.nonlinearity = nonlinearity
        if learnable_h0:
            self.h0 = nn.Parameter(torch.empty((self.hidden_size,), **factory_kwargs))
        else:
            self.register_parameter('h0', None)
        self.reset_parameters()
        
    def _configure_output(self, **kwargs):
        """Fix this"""
        if kwargs.get('type', 'linear') == 'linear':
            readout = nn.Linear(self.hidden_size, self.output_size, **kwargs.get('params', {}))
        else:
            raise ValueError
        if kwargs.get('activation', 'none') == 'none':
            activation = nn.Identity()
        elif kwargs.get('activation', 'none') == 'softmax':
            activation = nn.Softmax(dim=-1)
        else:
            raise ValueError
        if kwargs.get('rate_readout', True):
            self.readout = nn.Sequential(
                self.rnn_cell.hfn,
                readout,
                activation
            )
        else:
            self.readout = nn.Sequential(
                readout,
                activation
            )
    
    def build_initial_state(self, batch_size, device=None, dtype=None):
        """Return B x H initial state tensor"""
        if self.h0 is None:
            hx = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        else:
            hx = self.h0.unsqueeze(0).expand(batch_size, -1)
        return hx