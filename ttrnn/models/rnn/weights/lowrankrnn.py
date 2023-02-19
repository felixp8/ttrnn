import torch
import torch.nn as nn
import math

from typing import Optional

from .base import WeightsBase
from ..connectivity.base import ConnectivityBase, ConnectivityStack


class lrRNNWeights(WeightsBase):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        rank: int,
        bias: bool = True, 
        init_config: dict = {}, 
        trainable_config: dict = {},
        connectivity: ConnectivityBase = ConnectivityStack(), 
        device=None, 
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight_config = {
            'weight_ih': (hidden_size, input_size),
            'weight_hh_m': (hidden_size, rank),
            'weight_hh_n': (hidden_size, rank),
            'bias': (hidden_size,) if bias else None,
        }
        if not init_config:
            # uniform_kwargs = {
            #     'a': -1.0 / math.sqrt(hidden_size), 
            #     'b': 1.0 / math.sqrt(hidden_size)
            # }
            init_config = {
                'weight_ih': ('normal_', {}),
                'weight_hh_m': ('normal_', {}),
                'weight_hh_n': ('normal_', {}),
                'bias': ('zeros_', {}),
            }
        super(lrRNNWeights, self).__init__(
            weight_config=weight_config,
            init_config=init_config,
            trainable_config=trainable_config,
            connectivity=connectivity,
            **factory_kwargs
        )
        self.weight_names.append('weight_hh') # ghost param
    
    def get(self, weight_name: str, cached: bool = False) -> torch.Tensor:
        if weight_name not in self.weight_names:
            raise ValueError(f"Weight {weight_name} not found")
        if cached:
            weight = self.cache.get(weight_name, None)
            if weight is not None:
                return weight
        if weight_name in ['weight_hh_m', 'weight_hh_n']:
            return getattr(self, weight_name) # don't apply connectivity to these
        elif weight_name == 'weight_hh':
            m = getattr(self, 'weight_hh_m')
            n = getattr(self, 'weight_hh_n')
            weight = torch.mm(m, n.t())
        else:
            weight = getattr(self, weight_name)
        weight = self.connectivity(**dict(weight_name=weight))[weight_name]
        return weight
    
    def get_weight_ih(self, cached: bool = False) -> torch.Tensor:
        return self.get('weight_ih', cached=cached)
    
    def get_weight_hh(self, cached: bool = False) -> torch.Tensor:
        return self.get('weight_hh', cached=cached)
    
    def get_bias(self, cached: bool = False) -> torch.Tensor:
        return self.get('bias', cached=cached)
    
    def forward(self, cached: bool = False) -> dict[str, torch.Tensor]:
        weights = {
            'weight_ih': getattr(self, 'weight_ih'),
            'weight_hh': torch.mm(getattr(self, 'weight_hh_m'), 
                getattr(self, 'weight_hh_n').T),
            'bias': getattr(self, 'bias'),
        }
        weights = self.connectivity(**weights)
        self.cache = weights
        return weights

    # def svd_reparametrization(self):
    #     """
    #     Orthogonalize m and n via SVD
    #     """
    #     with torch.no_grad():
    #         structure = (self.m @ self.n.t()).numpy()
    #         m, s, n = np.linalg.svd(structure, full_matrices=False)
    #         m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
    #         self.m.set_(torch.from_numpy(m * np.sqrt(s)))
    #         self.n.set_(torch.from_numpy(n.transpose() * np.sqrt(s)))
    #         self._define_proxy_parameters()