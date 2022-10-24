import numpy as np

# just use torch.optim for this
# class SGD:
#     def __init__(self, lr=1e-4):
#         super(SGD, self).__init__()
#         self.lr = lr

class Hebbian:
    def __init__(self, params, lr=1e-4, fr_history=-1, n_conditions=1, nonlinearity='cubic'):
        super(Hebbian, self).__init__()
        self.lr = lr
        self.fr_history = fr_history
        self.set_nonlinearity(nonlinearity)
        self.Ravg = np.zeros(n_conditions)
        self.p = params
    
    def step(self, R, alpha_R, states, activations):
        # states: (T x H)
        if self.fr_history < 0:
            means = np.cumsum(states, axis=0) / np.arange(states.shape[0])[:, None]
        elif self.fr_history == 0 or self.fr_history == 1:
            means = states
        else:
            kernel = np.ones(self.fr_history)
            divisor = np.where(
                np.arange(states.shape[0]) < self.fr_history, 
                np.arange(states.shape[0]), 
                np.ones(states.shape[0]) * self.fr_history
            )
            filt = lambda x: np.convolve(x, kernel, mode='full')[:-(self.fr_history - 1)] / divisor
            means = np.apply_along_axis(filt, axis=0, arr=states)
        dx = states - means # T x H
        activations = np.roll(activations, 1, axis=0)
        activations[0, :] = 0.
        eligibility_trace = self.S(activations.T @ dx)
        dJ = self.lr * eligibility_trace * (R - self.Ravg)
        dJ = np.clip(dJ, -1e-4, 1e-4)
        self.p = self.p - dJ
        self.Ravg = alpha_R * self.Ravg + (1 - alpha_R) * R

    def set_nonlinearity(self, nonlinearity):
        if (nonlinearity == 'cubic'):
            self.S = lambda x: np.power(x, 3)
        else:
            raise ValueError(f"{nonlinearity} is not a supported nonlinearity")
