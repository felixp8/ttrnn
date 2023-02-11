import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RNNBase, RNNCellBase
from .weights import LSTMWeights


class leakyLSTMCell(RNNCellBase):
    """Discretized LSTM
    WARNING: to my knowledge, no precedent in literature

    i_t = sigmoid(W_hi @ h_{t-1} + W_ii @ u_t + b_i)
    f_t = sigmoid(W_hf @ h_{t-1} + W_if @ u_t + b_f)
    g_t = tanh(W_hg @ h_{t-1} + W_ig @ u_t + b_g)
    o_t = sigmoid(W_ho @ h_{t-1} + W_io @ u_t + b_o)
    c_t = (1 - alpha) * f_t * c_{t-1} + alpha * i_t * g_t
    h_t = o_t * tanh(c_t)
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias=True, dt=10, tau=50, 
                 init_config={}, noise_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(leakyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = LSTMWeights(input_size=input_size, hidden_size=hidden_size, bias=bias, init_config=init_config, **factory_kwargs)
    
        self.use_noise = noise_kwargs.get('use_noise', False)
        if self.use_noise:
            self.noise_type = noise_kwargs.get('noise_type', 'normal')
            self.noise_params = noise_kwargs.get('noise_params', {})

        self.dt = dt
        self.tau = tau

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
        return self.dt / self.tau
    
    def reset_parameters(self):
        self.weights.reset_parameters()
        self.weights(cached=False)
    
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
        bias = weights['bias']
        device, dtype = input.device, input.dtype
        h, c = hx
        if self.bias is None:
            ifgo = torch.mm(h, weight_hh.t()) + torch.mm(input, weight_ih.t())
        else:
            ifgo = torch.mm(h, weight_hh.t()) + torch.mm(input, weight_ih.t()) + bias
        i, f, g, o = ifgo.chunk(4, 1)
        i = F.sigmoid(i)
        f = F.sigmoid(f)
        g = F.tanh(g + self.noise())
        o = F.sigmoid(o)
        cy = (1 - self.alpha) * (f * c) + self.alpha * (i * g)
        hy = o * torch.tanh(cy)
        return (hy, cy)

class leakyLSTM(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size, bias=True, 
                 dt=10, tau=50, learnable_h0=True, batch_first=False,
                 init_config={}, noise_kwargs={}, output_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = leakyLSTMCell(input_size, hidden_size, bias, dt, tau, init_config, noise_kwargs, device, dtype)
        super(leakyLSTM, self).__init__(rnn_cell, input_size, hidden_size, output_size, batch_first, output_kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        if learnable_h0:
            self.h0 = nn.Parameter(torch.empty((self.hidden_size * 2,), **factory_kwargs))
        else:
            self.register_parameter('h0', None)
            # self.h0 = nn.Parameter(torch.zeros((1, self.hidden_size), **factory_kwargs), requires_grad=False)
        self.reset_parameters()
    
    def build_initial_state(self, batch_size, device=None, dtype=None):
        """Return B x H initial state tensor"""
        if self.h0 is None:
            zeros = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
            hx = (zeros, zeros)
        else:
            hx = self.h0.unsqueeze(0).expand(batch_size, -1).chunk(2, 1)
        return hx