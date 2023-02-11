import torch
import torch.nn as nn
import math

from typing import Optional

from .base import WeightsBase
from ..connectivity.base import ConnectivityBase, ConnectivityStack


class RNNWeights(WeightsBase):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bias: bool = True, 
        init_dict: Optional[dict] = None, 
        connectivity: ConnectivityBase = ConnectivityStack, 
        device=None, 
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight_dict = {
            'weight_ih': (hidden_size, input_size),
            'weight_hh': (hidden_size, hidden_size),
            'bias': (hidden_size,) if bias else None,
        }
        if init_dict is None:
            uniform_kwargs = {
                'a': -1.0 / math.sqrt(hidden_size), 
                'b': 1.0 / math.sqrt(hidden_size)
            }
            init_dict = {
                'weight_ih': ('uniform_', uniform_kwargs),
                'weight_hh': ('uniform_', uniform_kwargs),
                'bias': ('uniform_', uniform_kwargs),
            }
        super(RNNWeights, self).__init__(
            weight_dict=weight_dict,
            init_dict=init_dict,
            connectivity=connectivity,
            **factory_kwargs
        )
    
    def get_weight_ih(self) -> torch.Tensor:
        return self.get('weight_ih')
    
    def get_weight_hh(self) -> torch.Tensor:
        return self.get('weight_hh')
    
    def get_bias(self) -> torch.Tensor:
        return self.get('bias')