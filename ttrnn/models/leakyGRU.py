import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RNNBase, RNNCellBase
from .weights import GRUWeights


class leakyGRUCell(RNNCellBase):
    """Discretized GRU

    r_t = sigmoid(W_hr @ h_{t-1} + W_ir @ u_t + b_r)
    z_t = sigmoid(W_hz @ h_{t-1} + W_iz @ u_t + b_z)
    n_t = tanh(r_t * (W_hn @ h_{t-1} + b_hn) + W_in @ u_t + b_in)
    h_t = z_t * alpha * n_t + (1 - z_t * alpha) * h_{t-1}
    y_t = g(W_out @ h_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias=True, dt=10, tau=50, 
                 init_config={}, noise_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(leakyGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = GRUWeights(input_size=input_size, hidden_size=hidden_size, bias=bias, init_config=init_config, **factory_kwargs)
    
        self.use_noise = noise_kwargs.get('use_noise', False)
        if self.use_noise:
            self.noise_type = noise_kwargs.get('noise_type', 'normal')
            self.noise_params = noise_kwargs.get('noise_params', {})

        self.dt = dt
        self.tau = tau

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.reset_parameters()
        self.weights(cached=False)

    @property
    def weight_ih(self):
        return self.weights.get_weight_ih(cached=True)
    
    @property
    def weight_hh(self):
        return self.weights.get_weight_hh(cached=True)
    
    @property
    def bias_ih(self):
        return self.weights.get_bias_ih(cached=True)
        
    @property
    def bias_hh(self):
        return self.weights.get_bias_hh(cached=True)
    
    @property
    def alpha(self):
        return self.dt / self.tau
    
    def set_noise(self, use_noise=None, noise_kwargs={}):
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
        bias_ih = weights['bias_ih']
        bias_hh = weights['bias_hh']
        device, dtype = input.device, input.dtype
        if self.bias_hh is None:
            rznh = torch.mm(hx, weight_hh.t())
            rznu = torch.mm(input, weight_ih.t())
        else:
            rznh = torch.mm(hx, weight_hh.t()) + bias_hh
            rznu = torch.mm(input, weight_ih.t()) + bias_ih
        rh, zh, nh = rznh.chunk(3, 1)
        ru, zu, nu = rznu.chunk(3, 1)
        r = F.sigmoid(rh + ru)
        z = F.sigmoid(zh + zu)
        n = F.tanh(r * nh + nu + self.noise(device, dtype))
        hx = hx * (1 - self.alpha * z) + n * self.alpha * z
        return hx

class leakyGRU(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size, bias=True, 
                 dt=10, tau=50, learnable_h0=True, batch_first=False,
                 init_config={}, noise_kwargs={}, output_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = leakyGRUCell(input_size, hidden_size, bias, dt, tau, init_config, noise_kwargs, device, dtype)
        super(leakyGRU, self).__init__(rnn_cell, input_size, hidden_size, output_size, batch_first, output_kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        if learnable_h0:
            self.h0 = nn.Parameter(torch.empty((self.hidden_size,), **factory_kwargs))
        else:
            self.register_parameter('h0', None)
            # self.h0 = nn.Parameter(torch.zeros((1, self.hidden_size), **factory_kwargs), requires_grad=False)
        self.reset_parameters()
    
    def build_initial_state(self, batch_size, device=None, dtype=None):
        """Return B x H initial state tensor"""
        if self.h0 is None:
            hx = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        else:
            hx = self.h0.unsqueeze(0).expand(batch_size, -1)
        return hx