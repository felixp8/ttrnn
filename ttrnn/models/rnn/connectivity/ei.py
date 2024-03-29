import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

from .base import ConnectivityBase


class EIConnectivity(ConnectivityBase):
    """Excitatory-inhibitory"""
    supported = ['RNN', 'leakyRNN', 'rateRNN'] # TODO: add checks for this

    def __init__(
        self, 
        inhibitory_frac: float = 0.2, 
        weight_dict: dict = {}, 
        process_weight_dict: dict = {}, 
        rectifier: Literal["abs", "relu"] = "abs",
        enforce_dale: bool = True,
        device=None, 
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(EIConnectivity, self).__init__(
            process_weight_dict=process_weight_dict,
            **factory_kwargs
        )

        self.inhibitory_frac = inhibitory_frac
        if rectifier == "abs":
            self.rectifier = torch.abs
        else:
            self.rectifier = F.relu
        
        for weight_name, weight_size in weight_dict.items():
            if self.process_weight_dict.get(weight_name, False):
                sign = nn.Parameter( # TODO: follow Dale's law (each column has same sign)
                    torch.sign(torch.rand(weight_size, **factory_kwargs) - self.inhibitory_frac),
                    requires_grad=False,
                )
                setattr(self, weight_name + '_sign', sign)
        
    def forward(self, **kwargs):
        weights = {}
        for weight_name, weight in kwargs.items():
            if self.process_weight_dict.get(weight_name, False):
                weights[weight_name] = (self.rectifier(weight) *
                    getattr(self, weight_name + '_sign'))
            else:
                weights[weight_name] = weight
        return weights
