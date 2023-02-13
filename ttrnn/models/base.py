import torch
import torch.nn as nn

from typing import Optional

from .rnn import RNNBase


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()


class TaskRNN(ModelBase):
    def __init__(
        self,
        rnn: RNNBase,
        encoder: nn.Module = nn.Identity(),
        
    )