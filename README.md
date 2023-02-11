# ttrnn
PyTorch package for task-trained RNNs

## Goals
### Models
**Implemented**
* Vanilla RNN (Elman et al. 1990)
* GRU (Cho et al. 2014)
* LSTM (Hochreiter et al. 1997)
* Low-rank w/ no noise (Mastrogiuseppe 2017) - *untested*

**To do**
* Short-term plasticity (Masse et al. 2018)
* Low-rank w/ fixed noise (Mastrogiuseppe 2017)
* Support low-rank, mixture-of-Gaussian bases (Beiran et al. 2021)
* UGRNN (Collins et al. 2017)
* Multiplicative RNN (Sustkever et al. 2011)
* Others? euRNN, reservoir models, spiking RNNs, etc...

### Connectivity constraints
**To do**
* Excitatory-Inhibitory / Dale's law (Song et al. 2016)
* Arbitrary sparsity
* Locality-masked RNNs (Khona et al. 2022)

### Auxiliary losses
**To do**
* L1 / L2
* Minimum description length
* Spatial embedding (Achterberg et al. 2022)
* Network communicability (Achterberg et al. 2022)
* Slow point speed (Haviv et al. 2019)

### Learning Rules
**Implemented (native in Torch)**
* SGD
* Adam

**To do**
* SGD with clipping & scaling (Pascanu et al. 2013)
* Hessian-free (Martens et al. 2011)
* RTRL (Williams et al. 1989)
* Recursive least squares (Haykins 2002)
* Hebbian (Miconi 2017)
* RFLO (Murray et al. 2019)
* etc...

### Analysis
**Implemented (kind of)**
* Fixed-point finding
* Tensorboard / WandB logging during training

**To do**
* JSLDS co-training
* RADD / other distillation approaches
* Neuron clustering / task variance
* etc...

### Misc
**Implemented (kind of, supported in NeuroGym)**
* Multi-task training, mixed or sequential

**To do**
* Flexible support for different loss functions
* Reinforcement learning & meta-reinforcement learning
* etc...

## Inspirations
* Neurogym
* psychRNN
* [Maheswaranathan et al. 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7416639/)
* [McMahan et al. 2021](https://proceedings.neurips.cc/paper/2021/hash/b87039703fe79778e9f140b78621d7fb-Abstract.html)