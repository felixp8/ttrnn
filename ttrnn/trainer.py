import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from .models import RNN

class RNNTrainer(pl.LightningModule):
    def __init__(self, rnn_type='rnn', rnn_params={}, optim_type='sgd', optim_params={}, loss_func='mse_loss'):
        super(RNNTrainer, self).__init__()
        self.save_hyperparameters()
        # self.build_rnn(rnn_type, rnn_params)
        self.build_rnn()
        self.set_loss_func()

    def build_rnn(self):
        rnn_type = self.hparams.get('rnn_type', 'rnn')
        rnn_params = self.hparams.get('rnn_params', {})
        if (rnn_type.lower() in ['rnn', 'vanilla', 'elman']):
            self.model = RNN(**rnn_params)
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
        else:
            raise ValueError()
        return {
            'optimizer': optimizer
        }
    
    def forward(self, X, hx=None):
        outputs, (hs, rs) = self.model(X, hx)
        return outputs, (hs, rs)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, (hs, rs) = self.forward(inputs)
        loss = self.loss_func(outputs, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, (hs, rs) = self.forward(inputs)
        loss = self.loss_func(outputs, targets)
        self.log("valid_loss", loss)
        return loss

    def set_loss_func(self):
        loss_func_name = self.hparams.get('loss_func', 'mse')
        if (loss_func_name.lower() == 'mse_loss'):
            self.loss_func = F.mse_loss
        elif (loss_func_name.lower() == 'cross_entropy'):
            raise ValueError() # NOT WORKING - array shapes issue
            # self.loss_func = F.cross_entropy
        else:
            raise ValueError()
