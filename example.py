import torch
import pytorch_lightning as pl
# import neurogym as ngym
from ttrnn.trainer import SupervisedRNN
from ttrnn.dataset import NeurogymTaskDataset, NeurogymDataLoader, DiscreteToBoxWrapper
from ttrnn.callbacks import SuccessRate, TrajectoryPlot, TaskPlot

rnn_params = {
    'input_size': 3, 
    'hidden_size': 32, 
    'output_size': 3, 
    'nonlinearity': 'relu',
    'bias': False, 
    'learnable_h0': False,
    'batch_first': True,
    'init_kwargs': {'init_func': 'normal_', 'kwargs': {'mean': 0.0, 'std': 1 / (32 ** 0.5)}},
    'output_kwargs': {'type': 'linear', 'activation': 'softmax'},
}

model = SupervisedRNN(
    rnn_type='rnn',
    rnn_params=rnn_params,
    optim_type='adam',
    optim_params={'lr': 1e-4},
    loss_func='cross_entropy',
)

task = 'PerceptualDecisionMaking-v0'
env_kwargs = {'dt': 50, 'timing': {'fixation': 100, 'stimulus': 2000, 'delay': 0, 'decision': 400}}
wrappers = [] # [(DiscreteToBoxWrapper, {})]

train_dataloader = NeurogymDataLoader(
    NeurogymTaskDataset(task, env_kwargs, wrappers=wrappers, num_trials=100, seq_len=50, batch_first=True), 
    static=False,
    batch_size=1, 
    shuffle=True, 
)

val_dataloader = NeurogymDataLoader(
    NeurogymTaskDataset(task, env_kwargs, wrappers=wrappers, num_trials=20, seq_len=50, batch_first=True), 
    static=False,
    batch_size=20, 
    shuffle=False, 
)

loggers = [
    # pl.loggers.CSVLogger(save_dir="csv_logs"),
    # pl.loggers.WandbLogger(project='ttrnn-dev'),
]
callbacks = [
    # SuccessRate(log_every_n_epochs=2, include_abort=True, threshold=0.5),
    # TrajectoryPlot(log_every_n_epochs=2),
    # TaskPlot(log_every_n_epochs=2),
]

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=callbacks,
    gpus=0,
    num_nodes=1,
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=False,
    logger=loggers,
)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# import pdb; pdb.set_trace()