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
        output_size: Optional[int] = None,
        bias: bool = True, 
        init_config: Optional[dict] = None, 
        trainable_config: dict = {},
        connectivity: ConnectivityBase = ConnectivityStack(), 
        device=None, 
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight_config = {
            'weight_ih': (hidden_size, input_size),
            'weight_hh': (hidden_size, hidden_size),
            'bias_hh': (hidden_size,) if bias else None,
            'weight_ho': (output_size, hidden_size) if (output_size is not None) else None,
            'bias_ho': (output_size,) if (output_size is not None) else None,
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
                'weight_ho': ('uniform_', uniform_kwargs),
                'bias_ho': ('uniform_', uniform_kwargs),
            }
        super(RNNWeights, self).__init__(
            weight_config=weight_config,
            init_config=init_config,
            trainable_config=trainable_config,
            connectivity=connectivity,
            **factory_kwargs
        )
    
    def get_weight_ih(self, cached: bool = False) -> torch.Tensor:
        return self.get('weight_ih', cached=cached)
    
    def get_weight_hh(self, cached: bool = False) -> torch.Tensor:
        return self.get('weight_hh', cached=cached)
    
    def get_bias_hh(self, cached: bool = False) -> torch.Tensor:
        return self.get('bias_hh', cached=cached)

    def get_weight_ho(self, cached: bool = False) -> torch.Tensor:
        return self.get('weight_ho', cached=cached)

    def get_bias_ho(self, cached: bool = False) -> torch.Tensor:
        return self.get('bias_ho', cached=cached)