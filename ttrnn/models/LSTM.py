import numpy as np
import torch
import torch.nn as nn

from .base import RNNBase

class LSTM(RNNBase):
    """Standard LSTM

    i_t = sigmoid(W_hi @ h_{t-1} + W_ii @ u_t + b_i)
    f_t = sigmoid(W_hf @ h_{t-1} + W_if @ u_t + b_f)
    g_t = tanh(W_hg @ h_{t-1} + W_ig @ u_t + b_g)
    o_t = sigmoid(W_ho @ h_{t-1} + W_io @ u_t + b_o)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size, bias=True, nonlinearity='relu', 
                 learnable_h0=True, batch_first=False, output_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        rnn_cell = nn.LSTMCell(input_size, hidden_size, bias, device, dtype) # NOTE: does not support proj_size
        # readout = self.configure_output(**output_kwargs)
        super(LSTM, self).__init__(rnn_cell, input_size, hidden_size, output_size, batch_first, output_kwargs)
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