import torch
import torch.nn as nn

class SmallNeuralNetwork(nn.Module):
    """
    A lightweight fully connected neural network for testing weight quantization.
    
    Architecture:
      - Input: 784 (e.g., 28x28 image vector)
      - FC1: Linear(784, 128) -> ReLU
      - FC2: Linear(128, 64) -> ReLU
      - FC3: Linear(64, 10)
      
    Total Parameters:
      - FC1 weight: 784 * 128 = 100,352
      - FC1 bias:   128
      - FC2 weight: 128 * 64  = 8,192
      - FC2 bias:   64
      - FC3 weight: 64 * 10   = 640
      - FC3 bias:   10
      Total = 109,386 parameters (~437.5 KB FP32)
    """
    def __init__(self, input_dim: int = 784, hidden1: int = 128, hidden2: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def get_small_network(seed: int = 42) -> SmallNeuralNetwork:
    """Instantiates and returns a SmallNeuralNetwork with reproducible initialization."""
    torch.manual_seed(seed)
    model = SmallNeuralNetwork()
    return model
