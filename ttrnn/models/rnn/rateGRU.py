import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RNNBase, rateRNNCellBase
from .weights.gru import GRUWeights


class rateGRUCell(rateRNNCellBase):
    """Discretized rate GRU

    l_t = sigmoid(W_hr @ r_{t-1} + W_ir @ u_t + b_r)
    z_t = sigmoid(W_hz @ r_{t-1} + W_iz @ u_t + b_z)
    n_t = l_t * (W_hn @ r_{t-1} + b_hn) + W_in @ u_t + b_in
    h_t = z_t * alpha * n_t + (1 - z_t * alpha) * h_{t-1}
    r_t = f(h_t)
    y_t = g(W_out @ r_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'nonlinearity', 'bias']

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, nonlinearity='relu', 
                 dt=10, tau=50, init_config={}, noise_config={}, trainable_tau=False, 
                 rate_readout=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(rateGRUCell, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            dt=dt, 
            tau=tau, 
            trainable_tau=trainable_tau,
            noise_config=noise_config,
            rate_readout=rate_readout,
            **factory_kwargs
        )
        self.nonlinearity = nonlinearity
        self.weights = GRUWeights(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            bias=bias, 
            init_config=init_config, 
            **factory_kwargs,
        )

        if self.nonlinearity == 'relu': # for consistency with Torch
            self.hfn = nn.ReLU()
        elif self.nonlinearity == 'tanh': # for consistency with Torch
            self.hfn = nn.Tanh()
        elif hasattr(nn, self.nonlinearity):
            self.hfn = getattr(nn, self.nonlinearity)
        else:
            raise ValueError

        self.reset_parameters()

    @property
    def weight_ih(self):
        return self.weights.get_weight_ih(cached=False)
    
    @property
    def weight_hh(self):
        return self.weights.get_weight_hh(cached=False)
    
    @property
    def bias_ih(self):
        return self.weights.get_bias_ih(cached=False)
        
    @property
    def bias_hh(self):
        return self.weights.get_bias_hh(cached=False)
    
    @property
    def weight_ho(self):
        return self.weights.get_weight_ho(cached=False)
    
    @property
    def bias_ho(self):
        return self.weights.get_bias_ho(cached=False)
    
    def forward(self, input, hx, cached: bool = False):
        weights = self.weights(cached=cached)
        weight_ih = weights['weight_ih']
        weight_hh = weights['weight_hh']
        bias_ih = weights['bias_ih']
        bias_hh = weights['bias_hh']
        device, dtype = input.device, input.dtype
        if bias_hh is None:
            lznh = torch.mm(self.hfn(hx), weight_hh.t())
            lznu = torch.mm(input, weight_ih.t())
        else:
            lznh = torch.mm(self.hfn(hx), weight_hh.t()) + bias_hh
            lznu = torch.mm(input, weight_ih.t()) + bias_ih
        lh, zh, nh = lznh.chunk(3, 1)
        lu, zu, nu = lznu.chunk(3, 1)
        l = torch.sigmoid(lh + lu)
        z = torch.sigmoid(zh + zu)
        n = l * nh + nu + self.sample_noise()
        hx = hx * (1 - self.alpha * z) + n * self.alpha * z
        return hx


class rateGRU(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, nonlinearity='relu',
                 dt=10, tau=50, trainable_h0=True, batch_first=True,
                 init_config={}, noise_config={}, trainable_tau=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = rateGRUCell(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            bias=bias, 
            nonlinearity=nonlinearity, 
            dt=dt, 
            tau=tau, 
            init_config=init_config, 
            noise_config=noise_config, 
            trainable_tau=trainable_tau, 
            **factory_kwargs
        )
        super(rateGRU, self).__init__(
            rnn_cell=rnn_cell, 
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            batch_first=batch_first,
            trainable_h0=trainable_h0,
        )
        self.reset_parameters()