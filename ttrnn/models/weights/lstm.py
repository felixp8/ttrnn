import torch
import torch.nn as nn
import math

from typing import Optional

from .base import WeightsBase
from ..connectivity.base import ConnectivityBase, ConnectivityStack


class LSTMWeights(WeightsBase):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bias: bool = True, 
        init_config: Optional[dict] = None, 
        trainable_config: dict = {},
        connectivity: ConnectivityBase = ConnectivityStack(), 
        device=None, 
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight_config = {
            'weight_ih': (hidden_size*4, input_size),
            'weight_hh': (hidden_size*4, hidden_size),
            'bias': (hidden_size*4,) if bias else None,
        }
        if init_config is None:
            uniform_kwargs = {
                'a': -1.0 / math.sqrt(hidden_size), 
                'b': 1.0 / math.sqrt(hidden_size)
            }
            init_config = {
                'weight_ih': ('uniform_', uniform_kwargs),
                'weight_hh': ('uniform_', uniform_kwargs),
                'bias': ('uniform_', uniform_kwargs),
            }
        super(LSTMWeights, self).__init__(
            weight_config=weight_config,
            init_config=init_config,
            trainable_config=trainable_config,
            connectivity=connectivity,
            **factory_kwargs
        )
    
    def get_weight_ih(self):
        return self.get('weight_ih')
    
    def get_weight_hh(self):
        return self.get('weight_hh')
    
    def get_bias(self):
        return self.get('bias')