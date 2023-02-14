import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from .base import RNNBase, leakyRNNCellBase
from .weights import LSTMWeights


class leakyLSTMCell(leakyRNNCellBase):
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

    def __init__(self, input_size, hidden_size, proj_size=None, output_size=None,
                 bias=True, dt=10, tau=50, init_config={}, noise_config={}, trainable_tau=False, 
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(leakyLSTMCell, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            dt=dt,
            tau=tau,
            bias=bias,
            trainable_tau=trainable_tau,
            noise_config=noise_config,
            **factory_kwargs
        )
        self.weights = LSTMWeights(
            input_size=input_size, 
            hidden_size=hidden_size, 
            proj_size=proj_size,
            output_size=output_size,
            bias=bias, 
            init_config=init_config, 
            **factory_kwargs
        )
        if proj_size is not None:
            assert proj_size < hidden_size
            self.use_proj = True
        else:
            self.use_proj = False

        self.reset_parameters()

    @property
    def weight_ih(self):
        return self.weights.get_weight_ih(cached=True)
    
    @property
    def weight_hh(self):
        return self.weights.get_weight_hh(cached=True)
    
    @property
    def bias_hh(self):
        return self.weights.get_bias_hh(cached=True)
    
    @property
    def weight_hr(self):
        return self.weights.get_weight_hr(cached=True)
    
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
        bias = weights['bias_hh']
        device, dtype = input.device, input.dtype
        h, c = hx
        if bias is None:
            ifgo = torch.mm(h, weight_hh.t()) + torch.mm(input, weight_ih.t())
        else:
            ifgo = torch.mm(h, weight_hh.t()) + torch.mm(input, weight_ih.t()) + bias
        i, f, g, o = ifgo.chunk(4, 1)
        i = F.sigmoid(i)
        f = F.sigmoid(f)
        g = F.tanh(g + self.sample_noise(device, dtype))
        o = F.sigmoid(o)
        cy = (1 - self.alpha) * (f * c) + self.alpha * (i * g)
        hy = o * torch.tanh(cy)
        if self.use_proj:
            hy = torch.mm(hy, weights['weight_hr'].t())
        return (hy, cy)
    
    def output(self, hx: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h = hx[0]
        return super(leakyLSTMCell, self).output(h)


class leakyLSTM(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, proj_size=None, output_size=None, bias=True, 
                 dt=10, tau=50, trainable_h0=True, batch_first=True,
                 init_config={}, noise_config={}, trainable_tau=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = leakyLSTMCell(
            input_size=input_size, 
            hidden_size=hidden_size, 
            proj_size=proj_size,
            output_size=output_size,
            bias=bias, 
            dt=dt,
            tau=tau,
            init_config=init_config,
            noise_config=noise_config,
            trainable_tau=trainable_tau,
            **factory_kwargs
        ) 
        super(leakyLSTM, self).__init__(
            rnn_cell=rnn_cell, 
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            batch_first=batch_first,
            trainable_h0=trainable_h0,
            **factory_kwargs
        )
        self.reset_parameters()

    def init_initial_state(self, trainable: bool, device=None, dtype=None):
        if trainable:
            self.h0 = nn.Parameter(torch.empty((self.hidden_size * 2,), device=device, dtype=dtype))
        else:
            self.register_parameter('h0', None)
    
    def build_initial_state(self, batch_size, device=None, dtype=None):
        """Return B x H initial state tensor"""
        if self.h0 is None:
            zeros = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
            hx = (zeros, zeros)
        else:
            hx = self.h0.unsqueeze(0).expand(batch_size, -1).chunk(2, 1)
        return hx