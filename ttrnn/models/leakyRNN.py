import numpy as np
import torch
import torch.nn as nn

from .base import RNNBase, RNNCellBase

class leakyRNNCell(RNNCellBase):
    """Discretized Elman RNN

    h_t = (1 - alpha) * h_{t-1} + alpha * f(W_rec @ h_{t-1} + W_in @ u_t + b_x)
    y_t = g(W_out @ h_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'nonlinearity', 'bias']

    def __init__(self, input_size, hidden_size, nonlinearity='relu', dt=10, tau=50, 
                 bias=True, init_kwargs={}, noise_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(leakyRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.init_kwargs = init_kwargs
        self.weight_ih = nn.Parameter(torch.empty((hidden_size, input_size), **factory_kwargs))
        self.weight_hh = nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((1, hidden_size), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.set_nonlinearity()
        self.set_decay(dt, tau)
        self.reset_parameters()
        self.configure_noise(**noise_kwargs)
    
    def reset_parameters(self):
        init_func = getattr(nn.init, self.init_kwargs.get('init_func', 'normal_'))
        for weight in self.parameters():
            init_func(weight, **self.init_kwargs.get('kwargs', {}))
            
    def set_nonlinearity(self):
        if self.nonlinearity == 'relu':
            self.hfn = nn.ReLU()
        elif self.nonlinearity == 'tanh':
            self.hfn = nn.Tanh()
        else:
            raise ValueError
    
    def set_decay(self, dt=None, tau=None):
        if dt is not None:
            self.dt = dt
        if tau is not None:
            self.tau = tau
        self.alpha = self.dt / self.tau
    
    def configure_noise(self, **kwargs):
        self.use_noise = kwargs.get('use_noise', False)
        if self.use_noise:
            self.noise_type = kwargs.get('noise_type', 'normal')
            self.noise_params = kwargs.get('noise_params', {})
    
    def noise(self, device=None, dtype=None):
        if not self.use_noise:
            return 0. # torch.zeros((1, self.hidden_size), **self.factory_kwargs)
        else:
            out = torch.empty((1, self.hidden_size), device=device, dtype=dtype)
            return getattr(torch, self.noise_type)(size=(1, self.hidden_size), out=out, **self.noise_params)
    
    def forward(self, input, hx):
        device, dtype = input.device, input.dtype
        if self.bias is None:
            hx = hx * (1 - self.alpha) + \
                self.alpha * self.hfn(torch.mm(input, self.weight_ih.t()) + torch.mm(hx, self.weight_hh.t())) + self.noise(device, dtype)
        else:
            hx = hx * (1 - self.alpha) + \
                self.alpha * self.hfn(torch.mm(input, self.weight_ih.t()) + torch.mm(hx, self.weight_hh.t()) + self.bias) + self.noise(device, dtype)
        return hx

class leakyRNN(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size, bias=True, nonlinearity='relu', 
                 learnable_h0=True, batch_first=False, output_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = leakyRNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)
        readout = self.configure_output(**output_kwargs)
        super(leakyRNN, self).__init__(rnn_cell, readout, batch_first)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        if learnable_h0:
            self.h0 = nn.Parameter(torch.empty((1, self.hidden_size), **factory_kwargs))
        else:
            self.register_parameter('h0', None)
            # self.h0 = nn.Parameter(torch.zeros((1, self.hidden_size), **factory_kwargs), requires_grad=False)
        self.reset_parameters()
    
    def configure_output(self, **kwargs):
        if kwargs.get('type', 'linear') == 'linear':
            readout = nn.Linear(self.hidden_size, self.output_size)
        else:
            raise ValueError
        if kwargs.get('activation', 'none') == 'none':
            activation = nn.Identity()
        elif kwargs.get('activation', 'none') == 'softmax':
            activation = nn.Softmax(dim=-1)
        else:
            raise ValueError
        return nn.Sequential(
            readout,
            activation
        )
    
    def build_initial_state(self, batch_size, device=None, dtype=None):
        """Return B x H initial state tensor"""
        if self.h0 is None:
            hx = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        else:
            hx = self.h0.expand(batch_size, -1)
        return hx