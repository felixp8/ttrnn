import numpy as np
import torch
import torch.nn as nn

from .base import RNNBase, rateRNNCellBase
from .weights import RNNWeights


class rateRNNCell(rateRNNCellBase):
    """Discretized Rate RNN

    h_t = (1 - alpha) * h_{t-1} + alpha * (W_rec @ r_{t-1} + W_in @ u_t + b_x)
    r_t = f(h_t)
    o_t = g(W_out @ r_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'nonlinearity', 'bias']

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, nonlinearity='relu', 
                 dt=10, tau=50, init_config={}, noise_config={}, trainable_tau=False, 
                 rate_readout=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(rateRNNCell, self).__init__(
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
        self.weights = RNNWeights(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            bias=bias, 
            init_config=init_config, 
            **factory_kwargs
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
    def bias_hh(self):
        return self.weights.get_bias_hh(cached=False)
    
    @property
    def bias_ho(self):
        return self.weights.get_bias_ho(cached=False)
    
    def forward(self, input, hx, cached: bool = False):
        weights = self.weights(cached=cached)
        weight_ih = weights['weight_ih']
        weight_hh = weights['weight_hh']
        bias = weights['bias_hh']
        device, dtype = input.device, input.dtype
        if bias is None:
            hx = hx * (1 - self.alpha) + self.alpha * (
                torch.mm(input, weight_ih.t()) + 
                torch.mm(self.hfn(hx), weight_hh.t()) + 
                self.sample_noise(device, dtype))
        else:
            hx = hx * (1 - self.alpha) + self.alpha * (
                torch.mm(input, weight_ih.t()) + 
                torch.mm(self.hfn(hx), weight_hh.t()) + 
                bias + self.sample_noise(device, dtype))
        return hx


class rateRNN(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, nonlinearity='relu', 
                 dt=10, tau=50, trainable_h0=False, batch_first=True,
                 init_config={}, noise_config={}, trainable_tau=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = rateRNNCell(
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
        # readout = self.configure_output(rnn_cell=rnn_cell, hidden_size=hidden_size, output_size=output_size, **output_kwargs)
        super(rateRNN, self).__init__(
            rnn_cell=rnn_cell, 
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            batch_first=batch_first,
            trainable_h0=trainable_h0,
        )
        self.nonlinearity = nonlinearity
        self.reset_parameters()