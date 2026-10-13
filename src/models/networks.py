"""Small recurrent binary classifiers for streaming sequences."""

import torch
from torch import nn


class _RecurrentClassifier(nn.Module):
    recurrent_type = None

    def __init__(self, input_size, hidden_size=16):
        super().__init__()
        self.recurrent = self.recurrent_type(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, sequence):
        output, _ = self.recurrent(sequence)
        return self.classifier(output[:, -1, :])


class LSTMClassifier(_RecurrentClassifier):
    recurrent_type = nn.LSTM


class GRUClassifier(_RecurrentClassifier):
    recurrent_type = nn.GRU
