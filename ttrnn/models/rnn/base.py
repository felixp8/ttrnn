import numpy as np
import torch
import torch.nn as nn
import math

from typing import Union

from .weights import WeightsBase


class RNNCellBase(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = None

    def reset_parameters(self):
        self.weights.reset_parameters()
        self.weights(cached=False)
    
    def forward(self, input, hx):
        raise NotImplementedError
    
    def output(self, hx: torch.Tensor) -> torch.Tensor:
        weights = self.weights(cached=True)
        if weights['weight_ho'] is None:
            return hx
        return torch.mm(hx, weights['weight_ho'].t())


class leakyRNNCellBase(RNNCellBase):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int,
        dt: float, 
        tau: Union[int, float, np.ndarray, torch.Tensor], 
        bias: bool = True,
        trainable_tau: bool = False,
        tau_min: float = 1e-2,
        noise_config: dict = {},
        device=None, 
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(leakyRNNCellBase, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            **factory_kwargs
        )
        self.dt = dt

        if isinstance(tau, (int, float)):
            tau = torch.tensor([tau])
        elif isinstance(tau, np.ndarray):
            tau = torch.from_numpy(tau)

        tau = torch.where(
            tau > tau_min, 
            tau, 
            torch.full(tau.shape, tau_min)
        ).to(dtype).to(device)

        self.tau = nn.Parameter(tau, requires_grad=trainable_tau)
        self.trainable_tau = trainable_tau
        self.tau_min = tau_min
        if not self.trainable_tau:
            self.alpha_cached = self.dt / self.tau
        else:
            self.alpha_cached = None
        
        self.configure_noise(**noise_config)
    
    @property
    def alpha(self):
        if not self.trainable_tau:
            return self.alpha_cached
        tau_rect = torch.where(
            self.tau > self.tau_min, 
            self.tau, 
            torch.full(self.tau.shape, self.tau_min)
        )
        return self.dt / tau_rect
    
    def set_dt(self, dt: float):
        self.dt = dt
        if not self.trainable_tau:
            self.alpha_cached = self.dt / self.tau
        else:
            self.alpha_cached = None

    def configure_noise(self, enable: bool = False, noise_type: str = "", noise_params: dict = {}, **kwargs):
        if enable:
            assert noise_type != "" and hasattr(torch, noise_type)
        self.noise_enable = enable
        self.noise_type = noise_type
        self.noise_params = noise_params
    
    def disable_noise(self):
        self.noise_enable = False
    
    def enable_noise(self):
        self.noise_enable = True
        assert self.noise_type != "" and hasattr(torch, self.noise_type)
    
    def sample_noise(self, device=None, dtype=None):
        if not self.noise_enable:
            return 0. # torch.zeros((1, self.hidden_size), **self.factory_kwargs)
        else:
            out = torch.empty((1, self.hidden_size), device=device, dtype=dtype)
            return getattr(torch, self.noise_type)(
                size=(1, self.hidden_size), out=out, **self.noise_params)


class rateRNNCellBase(leakyRNNCellBase):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int,
        dt: float, 
        tau: Union[int, float, np.ndarray, torch.Tensor], 
        bias: bool = True,
        trainable_tau: bool = False,
        tau_min: float = 1e-2,
        noise_config: dict = {},
        rate_readout: bool = True,
        device=None, 
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(leakyRNNCellBase, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            dt=dt,
            tau=tau,
            trainable_tau=trainable_tau,
            tau_min=tau_min,
            noise_config=noise_config,
            **factory_kwargs
        )
        self.rate_readout = rate_readout
        self.hfn = None

    def output(self, hx: torch.Tensor) -> torch.Tensor:
        if self.rate_readout:
            hx = self.hfn(hx)
        weights = self.weights(cached=True)
        if weights['weight_ho'] is None:
            return hx
        return torch.mm(hx, weights['weight_ho'].t())


class RNNBase(nn.Module):
    """Base RNN class"""
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'batch_first']
    
    def __init__(self, rnn_cell, input_size, hidden_size, output_size, batch_first=True, 
                 trainable_h0=False, device=None, dtype=None):
        super(RNNBase, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.rnn_cell = rnn_cell
        self.batch_first = batch_first
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if trainable_h0:
            self.h0 = nn.Parameter(torch.empty((self.hidden_size,), **factory_kwargs))
        else:
            self.register_parameter('h0', None)

    def reset_parameters(self):
        """Set model weights"""
        if hasattr(self.rnn_cell, 'reset_parameters'):
            self.rnn_cell.reset_parameters()
        if self.h0 is not None:
            nn.init.uniform_( # TODO: allow configuring this
                self.h0, 
                a=-1.0 / math.sqrt(self.hidden_size), 
                b=1.0 / math.sqrt(self.hidden_size), 
            )

    def build_initial_state(self, batch_size, device=None, dtype=None):
        """Return B x H initial state tensor"""
        if self.h0 is None:
            hx = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        else:
            hx = self.h0.unsqueeze(0).expand(batch_size, -1)
        return hx
    
    def forward(self, input, hx=None):
        """Run RNN on sequence of inputs"""
        if hasattr(self.rnn_cell, 'weights') and \
            isinstance(getattr(self.rnn_cell, 'weights'), WeightsBase):
            self.rnn_cell.weights(cached=False)
        device, dtype = input.device, input.dtype
        if self.batch_first:
            batch_size, seq_len, input_size = input.shape
        else:
            seq_len, batch_size, input_size = input.shape
        if hx is None:
            hx = self.build_initial_state(batch_size, device, dtype)
        assert (self.input_size == input_size), f"Input size mismatch. Input is of size {input_size} " + \
            f"but RNN expected {self.input_size}"
        # assert (self.hidden_size == hx.shape[1])
        if self.batch_first:
            inputs = input.unbind(1)
        else:
            inputs = input.unbind(0)
        hs = []
        os = []
        for i in range(len(inputs)):
            o, hx = self.forward_step(inputs[i], hx)
            hs += [hx]
            os += [o]
        hs = self.concat_states(hs)
        os = torch.stack(os, dim=(1 if self.batch_first else 0))
        return os, hs
    
    def forward_step(self, input, hx):
        """Run RNN for single timestep"""
        hx = self.rnn_cell(input, hx)
        o = self.rnn_cell.output(hx)
        return o, hx

    def concat_states(self, hs):
        """Concatenate states along time dimension"""
        if isinstance(hs[0], tuple):
            hs = tuple([
                torch.stack(arrlist, dim=(1 if self.batch_first else 0)) for arrlist in zip(*hs)
            ])
        else:
            hs = torch.stack(hs, dim=(1 if self.batch_first else 0))
        return hs
    
    # def output(self, hx):
    #     """Compute output from current state"""
    #     if isinstance(hx, (list, tuple)):
    #         o = self.readout(hx[0])
    #     else:
    #         o = self.readout(hx)
    #     return o
