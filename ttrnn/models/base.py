import numpy as np
import torch
import torch.nn as nn

from .weights import WeightsBase


class RNNCellBase(nn.Module):
    def __init__(self):
        super(RNNCellBase, self).__init__()
    
    def forward(self, input, hx):
        raise NotImplementedError


class RNNBase(nn.Module):
    """Base RNN class"""
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'batch_first']
    
    def __init__(self, rnn_cell, input_size, hidden_size, output_size, batch_first=True, output_kwargs={}):
        super(RNNBase, self).__init__()
        self.rnn_cell = rnn_cell
        self.batch_first = batch_first
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._configure_output(**output_kwargs)

    def reset_parameters(self):
        """Set model weights"""
        if hasattr(self.rnn_cell, 'reset_parameters'):
            self.rnn_cell.reset_parameters()
        if hasattr(self.readout, 'reset_parameters'):
            self.readout.reset_parameters()
        else:
            if isinstance(self.readout, nn.Sequential):
                for layer in self.readout:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

    def _configure_output(self, **kwargs): # need to improve this
        if kwargs.get('type', 'linear') == 'linear':
            readout = nn.Linear(self.hidden_size, self.output_size, **kwargs.get('params', {}))
        else:
            readout = getattr(nn, kwargs.get('type'))(**kwargs.get('params'))
        if kwargs.get('activation', 'none') == 'none':
            activation = nn.Identity()
        elif kwargs.get('activation', 'none') == 'softmax':
            activation = nn.Softmax(dim=-1)
        else:
            raise ValueError
        self.readout = nn.Sequential(
            readout,
            activation
        )
    
    def build_initial_state(self, batch_size, device=None, dtype=None):
        """Return initial states. Override in sub-classes"""
        hx = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        return hx
    
    def forward(self, input, hx=None):
        """Run RNN on sequence of inputs"""
        if hasattr(self.rnn_cell, 'weights') and \
            isinstance(getattr(self.rnn_cell, 'weights'), WeightsBase):
            self.rnn_cell.weights(cached=False)
        device, dtype = input.device, input.dtype
        if self.batch_first:
            batch_size, seq_len, input_size = input.shape
        else:
            seq_len, batch_size, input_size = input.shape
        if hx is None:
            hx = self.build_initial_state(batch_size, device, dtype)
        assert (self.input_size == input_size), f"Input size mismatch. Input is of size {input_size} " + \
            f"but RNN expected {self.input_size}"
        # assert (self.hidden_size == hx.shape[1])
        if self.batch_first:
            inputs = input.unbind(1)
        else:
            inputs = input.unbind(0)
        hs = []
        os = []
        for i in range(len(inputs)):
            o, hx = self.forward_step(inputs[i], hx)
            hs += [hx]
            os += [o]
        hs = self.concat_states(hs)
        os = torch.stack(os, dim=(1 if self.batch_first else 0))
        return os, hs
    
    def forward_step(self, input, hx):
        """Run RNN for single timestep"""
        hx = self.rnn_cell(input, hx)
        o = self.output(hx)
        return o, hx

    def concat_states(self, hs):
        """Concatenate states along time dimension"""
        if isinstance(hs[0], tuple):
            hs = tuple([
                torch.stack(arrlist, dim=(1 if self.batch_first else 0)) for arrlist in zip(*hs)
            ])
        else:
            hs = torch.stack(hs, dim=(1 if self.batch_first else 0))
        return hs
    
    def output(self, hx):
        """Compute output from current state"""
        if isinstance(hx, (list, tuple)):
            o = self.readout(hx[0])
        else:
            o = self.readout(hx)
        return o
