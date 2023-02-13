import torch
import torch.nn as nn

from typing import Optional

from .rnn import RNNBase


class A2C(nn.Module):
    def __init__(
        self,
        rnn: RNNBase,
        encoder: nn.Module = nn.Identity(),
    )