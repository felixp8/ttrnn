import numpy as np
import torch
import torch.nn as nn

# class RNNCellBase:
#     def __init__(self):
#         super(RNNCellBase, self).__init__()
    
#     def forward(self, X):
#         return X

# class RNNBase:
#     def __init__(self):
#         super(RNNBase, self).__init__()
    
#     def forward(self, X):
#         return X

class RNN(nn.Module):
    """Discretized Elman RNN

    h_t = (1 - alpha) * h_{t-1} + alpha * (W_rec @ r_{t-1} + W_in @ u_t + b_x)
    r_t = f(h_t)
    y_t = g(W_out @ r_t + b_y)
    """
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'nonlinearity', 'bias',
                     'batch_first', 'bidirectional', 'h0']

    def __init__(self, input_size, hidden_size, output_size, nonlinearity='relu', dt=10, tau=50, 
                 bias=True, learnable_h0=True, batch_first=False,
                 init_kwargs={}, noise_kwargs={}, output_kwargs={}, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.init_kwargs = init_kwargs
        self.weight_ih = nn.Parameter(torch.empty((hidden_size, input_size), **factory_kwargs))
        self.weight_hh = nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((1, hidden_size), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            # self.bias = nn.Parameter(torch.zeros((1, hidden_size), **factory_kwargs), requires_grad=False)
        if learnable_h0:
            self.h0 = nn.Parameter(torch.empty((1, self.hidden_size), **factory_kwargs))
        else:
            self.register_parameter('h0', None)
            # self.h0 = nn.Parameter(torch.zeros((1, self.hidden_size), **factory_kwargs), requires_grad=False)
        self.set_nonlinearity()
        self.set_decay(dt, tau)
        self.configure_output(**output_kwargs)
        # self.initialize_weights(**init_kwargs)
        self.reset_parameters()
        self.configure_noise(**noise_kwargs)
    
    def reset_parameters(self):
        init_func = getattr(nn.init, self.init_kwargs.get('init_func', 'normal_'))
        for weight in self.parameters():
            init_func(weight, **self.init_kwargs.get('kwargs', {}))
    
    def set_nonlinearity(self):
        if self.nonlinearity == 'relu':
            self.hfn = nn.ReLU()
        elif self.nonlinearity == 'tanh':
            self.hfn = nn.Tanh()
        else:
            raise ValueError
    
    def set_decay(self, dt=None, tau=None):
        if dt is not None:
            self.dt = dt
        if tau is not None:
            self.tau = tau
        self.alpha = self.dt / self.tau
    
    def configure_output(self, **kwargs):
        if kwargs.get('type', 'linear') == 'linear':
            readout = nn.Linear(self.hidden_size, self.output_size)
        else:
            raise ValueError
        if kwargs.get('activation', 'none') == 'none':
            activation = nn.Identity()
        elif kwargs.get('activation', 'none') == 'softmax':
            activation = nn.Softmax(dim=-1)
        else:
            raise ValueError
        self.readout = nn.Sequential(
            readout,
            activation
        )
    
    def configure_noise(self, **kwargs):
        self.use_noise = kwargs.get('use_noise', False)
        if self.use_noise:
            self.noise_type = kwargs.get('noise_type', 'normal')
            self.noise_params = kwargs.get('noise_params', {})
    
    def noise(self, device=None, dtype=None):
        if not self.use_noise:
            return 0. # torch.zeros((1, self.hidden_size), **self.factory_kwargs)
        else:
            out = torch.empty((1, self.hidden_size), device=device, dtype=dtype)
            return getattr(torch, self.noise_type)(size=(1, self.hidden_size), out=out, **self.noise_params)
    
    def forward_step(self, input, state):
        device, dtype = input.device, input.dtype
        h_last, r_last = state
        if self.bias is None:
            h = h_last * (1 - self.alpha) + \
                self.alpha * (torch.mm(input, self.weight_ih.t()) + torch.mm(r_last, self.weight_hh.t())) + self.noise(device, dtype)
        else:
            h = h_last * (1 - self.alpha) + \
                self.alpha * (torch.mm(input, self.weight_ih.t()) + torch.mm(r_last, self.weight_hh.t()) + self.bias) + self.noise(device, dtype)
        r = self.hfn(h)
        o = self.readout(r)
        return o, (h, r)
    
    def forward(self, input, hx=None):
        device, dtype = input.device, input.dtype
        batch_size, seq_len, input_size = input.shape
        assert (self.input_size == input_size), "Input size mismatch" # TODO: improve error msg
        if hx is None:
            if self.h0 is None:
                h = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
            else:
                h = self.h0.expand(batch_size, -1)
        else:
            h = hx # why do I do this
        r = self.hfn(h)
        if self.batch_first:
            inputs = input.unbind(1)
        else:
            inputs = input.unbind(0)
        hs = []
        rs = []
        os = []
        for i in range(len(inputs)):
            o, (h, r) = self.forward_step(inputs[i], (h, r))
            hs += [h]
            rs += [r]
            os += [o]
        hs = torch.stack(hs, dim=(1 if self.batch_first else 0))
        rs = torch.stack(rs, dim=(1 if self.batch_first else 0))
        os = torch.stack(os, dim=(1 if self.batch_first else 0))
        # os = self.readout(rs) # need output at each timestep for online learning so
        return os, (hs, rs)

class jordanRNN:
    """Jordan RNN
    
    x_t = f(W_rec y_{t-1} + W_in u_t + b_x)
    y_t = g(W_out x_t + b_y)
    """