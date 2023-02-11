import numpy as np
import torch
import torch.nn as nn
import torch._VF as _VF

from typing import Optional

from .base import RNNBase, RNNCellBase
from .weights import GRUWeights


class GRUCell(RNNCellBase):
    """Standard GRU

    r_t = sigmoid(W_hr @ h_{t-1} + W_ir @ u_t + b_r)
    z_t = sigmoid(W_hz @ h_{t-1} + W_iz @ u_t + b_z)
    n_t = tanh(r_t * (W_hn @ h_{t-1} + b_hn) + W_in @ u_t + b_in)
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    y_t = g(W_out @ h_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='relu', init_config={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.weights = GRUWeights(input_size=input_size, hidden_size=hidden_size, bias=bias, init_config=init_config, **factory_kwargs)

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
    
    def reset_parameters(self):
        self.weights.reset_parameters()
        self.weights(cached=False)
    
    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        weights = self.weights(cached=True)
        ## Below copied from torch
        assert input.dim() in (1, 2), \
            f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        # TODO: support other nonlinearities
        ret = _VF.gru_cell(
            input, hx,
            weights['weight_ih'], weights['weight_hh'],
            weights['bias_ih'], weights['bias_hh'],
        )

        if not is_batched:
            ret = ret.squeeze(0)

        return ret


class GRU(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size, bias=True, nonlinearity='relu', 
                 learnable_h0=True, batch_first=False, init_config={}, output_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = GRUCell(
            input_size=input_size, 
            hidden_size=hidden_size, 
            bias=bias, 
            init_config=init_config,
            device=device, 
            dtype=dtype
        )
        # readout = self.configure_output(**output_kwargs)
        super(GRU, self).__init__(rnn_cell, input_size, hidden_size, output_size, batch_first, output_kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity
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