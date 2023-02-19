import torch
import torch.nn as nn

from typing import Optional

from .rnn import RNNBase


class SupervisedModel(nn.Module):
    __constants__ = ['has_encoder', 'has_readout']

    def __init__(
        self,
        rnn: RNNBase,
        encoder: Optional[nn.Module] = None,
        readout: Optional[nn.Module] = None,
    ):
        super(SupervisedModel, self).__init__()
        self.rnn = rnn
        self.encoder = encoder
        self.readout = readout
        self.has_encoder = (encoder is not None)
        self.has_readout = (readout is not None)
    
    def forward(self, X, hx=None):
        if self.has_encoder:
            X = self.encoder(X)
        outputs, hs = self.rnn(X, hx)
        if self.has_readout:
            outputs = self.readout(outputs)
        return outputs, hs