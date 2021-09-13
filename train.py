import torch.optim as optim
import torch
import torch.nn as nn
import model


class Train:
    def __init__(self):
        # hyperparameters
        self.N_EPOCH = 1000
        self.N_STEPS_PER_EPOCH = 1000
        self.TRAINING_MINIBATCH_SIZE = 10
        self.N_STEPS_IN_A_TRAINING_DATA = 100
        self.N_PLACE_CELL = 256
        self.N_HEAD_DIR_CELL = 12
        self.N_INPUT = 3
        self.WIDTH_ENV = 2.2
        self.HEIGHT_ENV = 2.2
        self.LEARNING_RATE = 1e-5
        self.MOMENTUM = 0.9
        self.BOTTLENECK_DROPOUT_RATE = 0.5
        self.GRADIENT_CLIPPING = 1e-5
        self.LSTM_HIDDEN_LAYER_SIZE = 128
        self.BOTTLENECK_LAYER_SIZE = 256
        self.cross_entropy_loss_func = nn.CrossEntropyLoss()
        self.model = model.Grid_Net(input_size=self.N_INPUT, lstm_hidden_layer_size=self.LSTM_HIDDEN_LAYER_SIZE,
                                    dropout_rate=self.BOTTLENECK_DROPOUT_RATE,
                                    bottleneck_size=self.BOTTLENECK_LAYER_SIZE, place_cell_size=self.N_PLACE_CELL,
                                    head_dir_cell_size=self.N_HEAD_DIR_CELL)
        self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=self.LEARNING_RATE, alpha=0.9,
                                       momentum=self.MOMENTUM, eps=1e-10)

    def train(self):
        for epoch in range(self.N_EPOCH):
            # training mode
            self.model.train()
            for steps in range(self.N_STEPS_PER_EPOCH):
                self.optimizer.zero_grad()
                self.model.forward(inputs=None, place_cell_acts=None, head_dir_cell_acts=None)
                # backward propagation
                loss = self.get_loss(place_cell_pred=None, head_dir_cell_pred=None, place_cell_target=None,
                                     head_dir_cell_target=None)
                loss.backward()
                # gradient clipping to prevent exploding/vanishing gradient
                nn.utils.clip_grad_norm_(self.model.parameters(), self.GRADIENT_CLIPPING)
                # update parameter of model
                self.optimizer.step()
            # evaluation mode
            self.model.eval()

    # calculate loss based on cross entropy + negative log
    def get_loss(self, place_cell_pred, head_dir_cell_pred, place_cell_target, head_dir_cell_target):
        # calculating loss and mean the output
        place_cell_loss = self.cross_entropy_loss_func(place_cell_pred, place_cell_target)
        head_dir_cell_loss = self.cross_entropy_loss_func(head_dir_cell_pred, head_dir_cell_target)
        return torch.mean(place_cell_loss + head_dir_cell_loss)
