import torch
import torch.nn as nn
import math

from ..connectivity.base import ConnectivityBase, ConnectivityStack


class WeightsBase(nn.Module):
    def __init__(
        self, 
        weight_config: dict = {}, 
        init_config: dict = {}, 
        trainable_config: dict = {}, 
        connectivity: ConnectivityBase = ConnectivityStack(), 
        device=None, 
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightsBase, self).__init__()
        self.weight_names = list(weight_config.keys())
        self.size_config = weight_config
        self.init_config = init_config
        self.trainable_config = trainable_config
        self.connectivity = connectivity
        for weight_name, weight_size in weight_config.items():
            if weight_size is None:
                self.register_parameter(weight_name, None)
            else:
                setattr(
                    self, 
                    weight_name, 
                    nn.Parameter(
                        torch.empty(weight_size, **factory_kwargs), 
                        requires_grad=trainable_config.get(weight_name, True), 
                    ),
                )
        self.reset_cache()
        self.reset_parameters()
        self.forward()
    
    def reset_parameters(self) -> None:
        for weight_name in self.weight_names:
            init_func, init_kwargs = self.init_config.get(
                weight_name, 
                ('uniform_', {
                    'a': -1.0 / math.sqrt(getattr(self, weight_name).shape[0]),
                    'b': 1.0 / math.sqrt(getattr(self, weight_name).shape[0]),
                })
            )
            init_func = getattr(nn.init, init_func)
            init_func(getattr(self, weight_name), **init_kwargs)
    
    def get(self, weight_name: str, cached: bool = False) -> torch.Tensor:
        if weight_name not in self.weight_names:
            raise ValueError(f"Weight {weight_name} not found")
        if cached:
            weight = self.cache.get(weight_name, None)
            if weight is not None:
                return weight
        weight = getattr(self, weight_name)
        weight = self.connectivity(**dict(weight_name=weight))[weight_name]
        return weight
    
    def forward(self, cached: bool = False) -> dict[str, torch.Tensor]:
        if cached:
            if self.cache: # check if all weights in cache?
                return self.cache
        weights = dict(self.named_parameters())
        weights = self.connectivity(**weights)
        self.cache = weights
        return weights
    
    def reset_cache(self):
        self.cache = {}