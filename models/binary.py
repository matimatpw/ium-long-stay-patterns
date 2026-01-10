import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.0):
        """
        Flexible binary classifier with configurable architecture.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes (e.g., [64, 32])
            dropout_rate: Dropout probability (0.0 means no dropout)
        """
        super(BinaryClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
