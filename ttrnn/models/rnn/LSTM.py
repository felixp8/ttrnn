import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._VF as _VF

from typing import Optional, Tuple

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
    __constants__ = ['input_size', 'hidden_size', 'bias', 'use_proj']

    def __init__(self, input_size, hidden_size, proj_size=None, output_size=None, 
                 bias=True, init_config={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LSTMCell, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
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
        return self.weights.get_weight_ih(cached=False)
    
    @property
    def weight_hh(self):
        return self.weights.get_weight_hh(cached=False)
    
    @property
    def bias_hh(self):
        return self.weights.get_bias_hh(cached=False)
    
    @property
    def weight_hr(self):
        return self.weights.get_weight_hr(cached=False)
    
    @property
    def weight_ho(self):
        return self.weights.get_weight_ho(cached=False)

    @property
    def bias_ho(self):
        return self.weights.get_bias_ho(cached=False)
    
    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, cached: bool = False) -> torch.Tensor:
        weights = self.weights(cached=cached)
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
        if not self.use_proj:
            ret = _VF.lstm_cell(
                input, hx,
                weights['weight_ih'], weights['weight_hh'],
                weights['bias_hh'], None,
            )
        else:
            weight_ih = weights['weight_ih']
            weight_hh = weights['weight_hh']
            bias = weights['bias_hh']
            weight_hr = weights['weight_hr']
            h, c = hx
            if bias is None:
                ifgo = torch.mm(h, weight_hh.t()) + torch.mm(input, weight_ih.t())
            else:
                ifgo = torch.mm(h, weight_hh.t()) + torch.mm(input, weight_ih.t()) + bias
            i, f, g, o = ifgo.chunk(4, 1)
            i = F.sigmoid(i)
            f = F.sigmoid(f)
            g = F.tanh(g)
            o = F.sigmoid(o)
            cy = (f * c) + (i * g)
            hy = o * torch.tanh(cy)
            hy = torch.mm(hy, weight_hr.t())
            ret = (hy, cy)

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret
    
    def output(self, hx: Tuple[torch.Tensor, torch.Tensor], cached: bool = False) -> torch.Tensor:
        h = hx[0]
        return super(LSTMCell, self).output(h, cached=cached)


class LSTM(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, proj_size=None, output_size=None, bias=True, 
                 trainable_h0=True, batch_first=True, init_config={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = LSTMCell(
            input_size=input_size, 
            hidden_size=hidden_size, 
            proj_size=proj_size,
            output_size=output_size,
            bias=bias, 
            init_config=init_config,
            **factory_kwargs
        ) 
        super(LSTM, self).__init__(
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