import torch.optim as optim
import torch
import torch.nn as nn
# hyperparameters
N_EPOCH = 1000
N_STEPS_PER_EPOCH = 1000
TRAINING_MINIBATCH_SIZE = 10
N_STEPS_IN_A_TRAINING_DATA = 100
N_PLACE_CELL = 256
N_HEAD_DIR_CELL = 12
WIDTH_ENV = 2.2
HEIGHT_ENV = 2.2
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
BOTTLENECK_DROPOUT_RATE = 0.5
GRADIENT_CLIPPING = 1e-5

optimizer = optim.RMSprop(None,lr=LEARNING_RATE,alpha=0.9,momentum=MOMENTUM,eps=1e-10)
def initialization():
    pass
def train():
    pass