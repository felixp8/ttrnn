import torch
import pytorch_lightning as pl
# import neurogym as ngym
from ttrnn.trainer import RNNTrainer
from ttrnn.dataset import NeurogymTaskDataset, NeurogymDataLoader

rnn_params = {
    'input_size': 3, 
    'hidden_size': 32, 
    'output_size': 1, 
    'nonlinearity': 'relu',
    'bias': False, 
    'learnable_h0': False,
    'batch_first': True,
    'init_kwargs': {'init_func': 'normal_', 'kwargs': {'mean': 0.0, 'std': 1 / (32 ** 0.5)}},
    'output_kwargs': {'type': 'linear', 'activation': 'none'},
}

model = RNNTrainer(
    rnn_type='rnn',
    rnn_params=rnn_params,
    optim_type='sgd',
    optim_params={'lr': 1e-3},
    loss_func='mse_loss',
)

task = 'PerceptualDecisionMaking-v0'
env_kwargs = {'dt': 50}

train_dataloader = NeurogymDataLoader(
    NeurogymTaskDataset(task, env_kwargs, num_trials=300, seq_len=100, batch_first=True), 
    static=False,
    batch_size=1, 
    shuffle=True, 
)

valid_dataloader = NeurogymDataLoader(
    NeurogymTaskDataset(task, env_kwargs, num_trials=100, seq_len=100, batch_first=True), 
    static=False,
    batch_size=1, 
    shuffle=True, 
)

trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[],
    gpus=0,
    num_nodes=1,
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=False,
)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)