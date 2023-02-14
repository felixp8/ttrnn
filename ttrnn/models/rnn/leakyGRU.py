import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RNNBase, leakyRNNCellBase
from .weights import GRUWeights


class leakyGRUCell(leakyRNNCellBase):
    """Discretized GRU

    r_t = sigmoid(W_hr @ h_{t-1} + W_ir @ u_t + b_r)
    z_t = sigmoid(W_hz @ h_{t-1} + W_iz @ u_t + b_z)
    n_t = tanh(r_t * (W_hn @ h_{t-1} + b_hn) + W_in @ u_t + b_in)
    h_t = z_t * alpha * n_t + (1 - z_t * alpha) * h_{t-1}
    y_t = g(W_out @ h_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, 
                 dt=10, tau=50, init_config={}, noise_config={}, trainable_tau=False, 
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(leakyGRUCell, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            dt=dt,
            tau=tau,
            bias=bias,
            trainable_tau=trainable_tau,
            noise_config=noise_config,
            **factory_kwargs
        )
        self.weights = GRUWeights(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            bias=bias, 
            init_config=init_config, 
            **factory_kwargs
        )

        self.reset_parameters()

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
    def weight_ho(self):
        return self.weights.get_weight_ho(cached=True)
    
    @property
    def bias_ho(self):
        return self.weights.get_bias_ho(cached=True)
    
    def forward(self, input, hx):
        weights = self.weights(cached=True)
        weight_ih = weights['weight_ih']
        weight_hh = weights['weight_hh']
        bias_ih = weights['bias_ih']
        bias_hh = weights['bias_hh']
        device, dtype = input.device, input.dtype
        if bias_hh is None: # assume biases are paired
            rznh = torch.mm(hx, weight_hh.t())
            rznu = torch.mm(input, weight_ih.t())
        else:
            rznh = torch.mm(hx, weight_hh.t()) + bias_hh
            rznu = torch.mm(input, weight_ih.t()) + bias_ih
        rh, zh, nh = rznh.chunk(3, 1)
        ru, zu, nu = rznu.chunk(3, 1)
        r = F.sigmoid(rh + ru)
        z = F.sigmoid(zh + zu)
        n = F.tanh(r * nh + nu + self.sample_noise(device, dtype))
        hx = hx * (1 - self.alpha * z) + n * self.alpha * z
        return hx


class leakyGRU(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, 
                 dt=10, tau=50, trainable_h0=True, batch_first=True,
                 init_config={}, noise_config={}, trainable_tau=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = leakyGRUCell(
            input_size=input_size, 
            hidden_size=hidden_size,
            output_size=output_size,
            bias=bias, 
            dt=dt, 
            tau=tau, 
            init_config=init_config, 
            noise_config=noise_config, 
            trainable_tau=trainable_tau,
            **factory_kwargs
        )
        super(leakyGRU, self).__init__(
            rnn_cell=rnn_cell, 
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            batch_first=batch_first,
            trainable_h0=trainable_h0,
        )
        self.reset_parameters()