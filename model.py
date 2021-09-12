from torch import nn
import torch
import numpy as np


class Grid_Net(nn.Module):
    def __init__(self, input_size=3, lstm_hidden_layer_size=128, dropout_rate=0.5, bottleneck_size=256,
                 place_cell_size=256,
                 head_dir_cell_size=12):
        super(Grid_Net, self).__init__()

        # place/head dir cell activations -> hidden layer size of lstm
        self.init_lstm_cell = nn.Linear(place_cell_size + head_dir_cell_size, lstm_hidden_layer_size)
        self.init_lstm_state = nn.Linear(place_cell_size + head_dir_cell_size, lstm_hidden_layer_size)

        # LSTM for predicting the cell output
        self.rnn = nn.LSTM(input_size, lstm_hidden_layer_size)

        # bottleneck to cut down activation
        self.bottleneck = nn.Linear(lstm_hidden_layer_size, bottleneck_size)

        # automatically stops dropout when it is in eval mode
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax_place_cell = nn.Softmax()
        self.softmax_head_dir_cell = nn.Softmax()
        self.place_cell_pred = nn.Linear(bottleneck_size, place_cell_size)
        self.head_dir_cell_pred = nn.Linear(bottleneck_size, head_dir_cell_size)

    # input of velocity, cos and sin of angular velocity, and hidden state of cell activations
    def forward(self, inputs, place_cell_acts, head_dir_cell_acts):
        initial_condition = torch.cat(place_cell_acts, head_dir_cell_acts, dim=1)
        hidden_state = self.init_lstm_state(initial_condition)
        cell_state = self.init_lstm_cell(initial_condition)
        out, (hidden_state_out, cell_state_out) = self.rnn(inputs, (hidden_state, cell_state))
        out = self.bottleneck(out)
        out = self.dropout(out)
        place_cell_prediction = self.softmax_place_cell(self.place_cell_pred(out))
        head_dir_cell_prediction = self.softmax_head_dir_cell(self.head_dir_cell_pred(out))
        return place_cell_prediction, head_dir_cell_prediction, hidden_state_out, cell_state_out
# TODO: implement so that there is layers of stuff that can do backpropagation through time