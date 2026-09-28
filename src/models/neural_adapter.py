"""Online-learning adapter for the benchmark's recurrent classifiers."""

import numpy as np
import torch
from torch import nn

from .networks import GRUClassifier, LSTMClassifier


class NeuralAdapter:
    """Predict/update/reset interface shared by LSTM and GRU models."""

    ARCHITECTURES = {"lstm": LSTMClassifier, "gru": GRUClassifier}

    def __init__(self, architecture, input_size, seed=42, hidden_size=16,
                 learning_rate=0.01, sequence_length=10):
        architecture = architecture.lower()
        if architecture not in self.ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {architecture}")
        if input_size < 1:
            raise ValueError("input_size must be at least 1")
        self.architecture = architecture
        self.input_size = int(input_size)
        self.seed = int(seed)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.sequence_length = int(sequence_length)
        self.metadata = {
            "model_type": architecture.upper(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": 1,
            "sequence_length": self.sequence_length,
            "learning_rate": self.learning_rate,
            "optimizer": "Adam",
            "updates_per_label": 1,
            "device": "cpu",
            "seed": self.seed,
        }
        self.reset()

    def reset(self):
        # fork_rng restores caller RNG state, while the fixed seed makes
        # initialization repeatable without perturbing unrelated strategies.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = self.ARCHITECTURES[self.architecture](
                self.input_size, self.hidden_size
            ).cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def _tensor(self, sequence):
        values = np.asarray(sequence, dtype=np.float32)
        if values.shape != (self.sequence_length, self.input_size):
            raise ValueError(
                f"expected sequence shape {(self.sequence_length, self.input_size)}, "
                f"got {values.shape}"
            )
        return torch.from_numpy(values).unsqueeze(0)

    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._tensor(sequence))
        return int(torch.argmax(logits, dim=1).item())

    def update(self, sequence, label):
        self.model.train()
        x = self._tensor(sequence)
        y = torch.tensor([int(label)], dtype=torch.long)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model(x), y)
        loss.backward()
        self.optimizer.step()
