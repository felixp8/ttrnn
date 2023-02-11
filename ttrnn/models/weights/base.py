import torch
import torch.nn as nn
import math

from ..connectivity.base import ConnectivityBase, ConnectivityStack


class WeightsBase(nn.Module):
    def __init__(
        self, 
        weight_dict: dict = {}, 
        init_dict: dict = {}, 
        train_dict: dict = {}, 
        connectivity: ConnectivityBase = ConnectivityStack, 
        device=None, 
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightsBase, self).__init__()
        self.weight_dict = weight_dict
        self.init_dict = init_dict
        self.train_dict = train_dict
        self.connectivity = connectivity
        for weight_name, weight_size in weight_dict.items():
            if weight_size is None:
                self.register_parameter(weight_name, None)
            else:
                setattr(
                    self, 
                    weight_name, 
                    nn.Parameter(
                        torch.empty(weight_size), 
                        requires_grad=train_dict.get(weight_name, True), 
                        **factory_kwargs),
                )
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        for weight_name in self.weight_dict.keys():
            init_func, init_kwargs = self.init_dict.get(
                weight_name, 
                ('uniform_', {
                    'a': -1.0 / math.sqrt(getattr(self, weight_name).shape[0]),
                    'b': 1.0 / math.sqrt(getattr(self, weight_name).shape[0]),
                })
            )
            init_func = getattr(nn.init, init_func)
            init_func(getattr(self, weight_name), **init_kwargs)
    
    def get(self, weight_name: str) -> torch.Tensor:
        weight = getattr(self, weight_name)
        weight = self.connectivity(weight_name=weight)[weight_name]
        return weight
    
    def forward(self) -> dict[str, torch.Tensor]:
        weights = dict(self.named_parameters())
        weights = self.connectivity(**weights)
        return weights