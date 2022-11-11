# Traditional distillation:
#   Train a lower-d RNN to produce outputs of larger RNN from
#   same inputs. Will require "soft" cross-entropy loss with
#   not one-hot encoded target outputs
# RADD:
#   Train a lower-d RNN to match readout from larger RNN hidden
#   states. Can use PCA, original paper used "stimulus readout vector"
#   and "block readout vector". Need to read about task to understand 
#   what that means.