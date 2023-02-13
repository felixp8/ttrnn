import numpy as np
import torch
import torch.nn as nn
import torch._VF as _VF

from typing import Optional

from .base import RNNBase, RNNCellBase
from .weights import LSTMWeights


class LSTMCell(RNNCellBase):
    """Standard LSTM

    i_t = sigmoid(W_hi @ h_{t-1} + W_ii @ u_t + b_i)
    f_t = sigmoid(W_hf @ h_{t-1} + W_if @ u_t + b_f)
    g_t = tanh(W_hg @ h_{t-1} + W_ig @ u_t + b_g)
    o_t = sigmoid(W_ho @ h_{t-1} + W_io @ u_t + b_o)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias=True, init_config={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = LSTMWeights(input_size=input_size, hidden_size=hidden_size, bias=bias, init_config=init_config, **factory_kwargs)

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
    
    def reset_parameters(self):
        self.weights.reset_parameters()
        self.weights(cached=False)
    
    def forward(self, input: torch.Tensor, hx: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        weights = self.weights(cached=True)
        assert input.dim() in (1, 2), \
            f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        # TODO: support other nonlinearities
        ret = _VF.lstm_cell(
            input, hx,
            weights['weight_ih'], weights['weight_hh'],
            weights['bias'], None,
        )

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret


class LSTM(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size, bias=True, nonlinearity='relu', 
                 learnable_h0=True, batch_first=False, init_config={}, output_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = LSTMCell( # NOTE: does not support proj_size
            input_size=input_size, 
            hidden_size=hidden_size, 
            bias=bias, 
            init_config=init_config,
            device=device, 
            dtype=dtype,
        ) 
        # readout = self.configure_output(**output_kwargs)
        super(LSTM, self).__init__(rnn_cell, input_size, hidden_size, output_size, batch_first, output_kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity
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