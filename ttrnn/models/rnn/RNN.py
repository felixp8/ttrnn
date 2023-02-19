import numpy as np
import torch
import torch.nn as nn
import torch._VF as _VF

from typing import Optional

from .base import RNNBase, RNNCellBase
from .weights import RNNWeights


class RNNCell(RNNCellBase):
    """Standard vanilla RNN

    h_t = f(W_rec @ h_{t-1} + W_in @ u_t + b_h)
    y_t = g(W_out @ h_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'nonlinearity', 'bias']

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, nonlinearity='relu', 
                 init_config={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RNNCell, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
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
    def weight_ho(self):
        return self.weights.get_weight_ho(cached=False)
    
    @property
    def bias_ho(self):
        return self.weights.get_bias_ho(cached=False)
    
    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None, cached: bool = False) -> torch.Tensor:
        weights = self.weights(cached=cached)
        ## Below copied from torch
        assert input.dim() in (1, 2), \
            f"RNNCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        if self.nonlinearity == "tanh":
            ret = _VF.rnn_tanh_cell(
                input, hx,
                weights['weight_ih'], weights['weight_hh'],
                weights['bias_hh'], None,
            )
        elif self.nonlinearity == "relu":
            # import pdb; pdb.set_trace()
            ret = _VF.rnn_relu_cell(
                input, hx,
                weights['weight_ih'], weights['weight_hh'],
                weights['bias_hh'], None,
            )
        else:
            ret = input  # TODO: support other nonlinearities
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        if not is_batched:
            ret = ret.squeeze(0)

        return ret


class RNN(RNNBase):
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size=None, bias=True, nonlinearity='relu', 
                 trainable_h0=False, batch_first=True, init_config={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = RNNCell(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            bias=bias, 
            nonlinearity=nonlinearity, 
            init_config=init_config,
            **factory_kwargs
        )
        # readout = self.configure_output(**output_kwargs)
        super(RNN, self).__init__(
            rnn_cell=rnn_cell, 
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            batch_first=batch_first,
            trainable_h0=trainable_h0,
        )
        self.nonlinearity = nonlinearity
        self.reset_parameters()