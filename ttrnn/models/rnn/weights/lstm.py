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
        proj_size: Optional[int] = None,
        output_size: Optional[int] = None,
        bias: bool = True, 
        init_config: Optional[dict] = None, 
        trainable_config: dict = {},
        connectivity: ConnectivityBase = ConnectivityStack(), 
        device=None, 
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        real_hidden_size = hidden_size if proj_size is None else proj_size
        weight_config = {
            'weight_ih': (hidden_size*4, input_size),
            'weight_hh': (hidden_size*4, real_hidden_size),
            'bias': (hidden_size*4,) if bias else None,
            'weight_hr': (real_hidden_size, hidden_size) if (proj_size is not None) else None,
            'weight_ho': (output_size, real_hidden_size) if (output_size is not None) else None,
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
                'weight_hr': ('uniform_', uniform_kwargs),
                'weight_ho': ('uniform_', uniform_kwargs),
            }
        super(LSTMWeights, self).__init__(
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
    
    def get_bias(self, cached: bool = False) -> torch.Tensor:
        return self.get('bias', cached=cached)
    
    def get_weight_hr(self, cached: bool = False) -> torch.Tensor:
        return self.get('weight_hr', cached=cached)
    
    def get_weight_ho(self, cached: bool = False) -> torch.Tensor:
        return self.get('weight_ho', cached=cached)
