import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        """
        Inicjalizacja z wykorzystaniem nn.Sequential.
        """
        super(BinaryClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Teraz forward po prostu przekazuje dane do zdefiniowanej metody 'network'.
        """
        return self.network(x)
