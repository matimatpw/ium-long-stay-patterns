import torch
import torch.nn as nn

class NaiveZeroClassifier(nn.Module):
    def __init__(self):
        super(NaiveZeroClassifier, self).__init__()
        # Model naiwny nie ma wag do trenowania

    def forward(self, x):
        """
        Zwraca tensor zer o takim samym rozmiarze partii (batch size) jak wejście.
        Wyjście ma kształt (batch_size, 1).
        """
        batch_size = x.size(0)
        # Tworzy tensor zer na tym samym urządzeniu (CPU/GPU) co dane wejściowe
        return torch.zeros((batch_size, 1), device=x.device)
