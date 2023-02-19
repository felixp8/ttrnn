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

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, 
                 init_config={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GRUCell, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
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
    
    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None, cached: bool = False) -> torch.Tensor:
        weights = self.weights(cached=cached)
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

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, 
                 trainable_h0=True, batch_first=True, init_config={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = GRUCell(
            input_size=input_size, 
            hidden_size=hidden_size,
            output_size=output_size, 
            bias=bias, 
            init_config=init_config,
            **factory_kwargs
        )
        super(GRU, self).__init__(
            rnn_cell=rnn_cell, 
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            batch_first=batch_first,
            trainable_h0=trainable_h0,
        )
        self.reset_parameters()