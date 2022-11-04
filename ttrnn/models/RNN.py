import numpy as np
import torch
import torch.nn as nn

from .base import RNNBase

class RNN(RNNBase):
    """Standard vanilla RNN

    h_t = f(W_rec @ h_{t-1} + W_in @ u_t + b_x)
    y_t = g(W_out @ h_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size, bias=True, nonlinearity='relu', 
                 learnable_h0=True, batch_first=False, output_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = nn.RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)
        # readout = self.configure_output(**output_kwargs)
        super(RNN, self).__init__(rnn_cell, input_size, hidden_size, output_size, batch_first, output_kwargs)
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
       
    def build_initial_state(self, batch_size, device=None, dtype=None):
        """Return B x H initial state tensor"""
        if self.h0 is None:
            hx = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        else:
            hx = self.h0.expand(batch_size, -1)
        return hx