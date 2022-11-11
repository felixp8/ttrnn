import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from . import models

def cross_entropy(input, target):
    """Wrapper for torch F.cross_entropy with input shape handling
    """
    # import pdb; pdb.set_trace()
    if len(input.shape) > 2: # Batch x Time x ...
        input = input.flatten(start_dim=0, end_dim=-2)
        target = target.flatten(start_dim=0, end_dim=-2)
    if (input.shape == target.shape): # Target is probabilities
        return F.cross_entropy(input, target)
    elif (len(target.shape) == 1): # Target is class label
        if (target.dtype != torch.long):
            target = target.to(torch.long)
        return F.cross_entropy(input, target)
    elif (target.shape[-1] == 1): # Target is expanded class label
        target = target.squeeze(-1)
        if (target.dtype != torch.long):
            target = target.to(torch.long)
        return F.cross_entropy(input, target)
    else:
        raise ValueError()

class SupervisedRNN(pl.LightningModule):
    def __init__(self, rnn_type='rnn', rnn_params={}, optim_type='sgd', optim_params={}, loss_func='mse_loss'):
        super(SupervisedRNN, self).__init__()
        self.save_hyperparameters()
        # self.build_rnn(rnn_type, rnn_params)
        self.build_rnn()
        self.set_loss_func()

    def build_rnn(self):
        rnn_type = self.hparams.get('rnn_type', 'RNN')
        rnn_params = self.hparams.get('rnn_params', {})
        if hasattr(models, rnn_type):
            self.model = getattr(models, rnn_type)(**rnn_params)
        else:
            raise ValueError()
    
    def configure_optimizers(self):
        optim_type = self.hparams.get('optim_type', 'sgd')
        optim_params = self.hparams.get('optim_params', {})
        # optimizer = self.get_optimizer(optim_type, optim_params)
        if (optim_type.lower() == 'sgd'):
            optimizer = optim.SGD(self.parameters(), **optim_params)
        elif (optim_type.lower() == 'adam'):
            optimizer = optim.Adam(self.parameters(), **optim_params)
        elif hasattr(optim, optim_type):
            optimizer = getattr(optim, optim_type)(self.parameters(), **optim_params)
        else:
            raise ValueError()
        return {
            'optimizer': optimizer,
        }
    
    def forward(self, X, hx=None):
        outputs, hs = self.model(X, hx)
        return outputs, hs
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, hs = self.forward(inputs)
        loss = self.loss_func(outputs, targets)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, hs = self.forward(inputs)
        loss = self.loss_func(outputs, targets)
        self.log("val/loss", loss)
        return loss

    def set_loss_func(self):
        loss_func_name = self.hparams.get('loss_func', 'mse_loss')
        if (loss_func_name.lower() == 'mse_loss'):
            self.loss_func = F.mse_loss
        elif (loss_func_name.lower() == 'cross_entropy'):
            # self.loss_func = F.cross_entropy
            self.loss_func = cross_entropy
        elif callable(loss_func_name):
            self.loss_func = loss_func_name
        else:
            raise ValueError()
