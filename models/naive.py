import torch
import torch.nn as nn

class NaiveZeroClassifier(nn.Module):
    def __init__(self):
        super(NaiveZeroClassifier, self).__init__()

    def forward(self, x):
        """
        Zwraca tensor zer o takim samym rozmiarze partii (batch size) jak wejście.
        Wyjście ma kształt (batch_size, 1).
        """
        batch_size = x.size(0)
        return torch.zeros((batch_size, 1), device=x.device)
