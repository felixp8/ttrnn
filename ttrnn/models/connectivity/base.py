import torch
import torch.nn as nn


class ConnectivityBase(nn.Module):
    def __init__(self, process_weight_dict={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConnectivityBase, self).__init__()
        self.process_weight_dict = process_weight_dict
    
    def forward(self, **kwargs):
        weights = {}
        for weight_name in self.process_weight_dict.keys():
            if kwargs.get(weight_name, None) is not None:
                weights[weight_name] = kwargs[weight_name]
        return weights


class ConnectivityStack(ConnectivityBase):
    def __init__(self, connectivity_list=[], device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConnectivityStack, self).__init__()
        self.connectivity_list = nn.ModuleList(connectivity_list)
    
    def forward(self, **kwargs):
        for layer in self.connectivity_list:
            kwargs = layer(**kwargs)
        return kwargs