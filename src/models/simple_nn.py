"""
Simple neural network model for classification tasks.
"""
import torch
import torch.nn as nn
from typing import Optional


class SimpleModel(nn.Module):
    """
    Simple feedforward neural network for classification.
    
    This model consists of two fully connected layers with ReLU activation
    and optional dropout for regularization.
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        num_classes: int = 10,
        dropout: float = 0.2
    ):
        """
        Initialize the model.
        
        Args:
            input_size: Size of input features.
            hidden_size: Size of hidden layer.
            num_classes: Number of output classes.
            dropout: Dropout probability for regularization.
        """
        super(SimpleModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, *) where * is flattened to input_size.
        
        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.fc2(x)
        return x
    
    def get_num_parameters(self) -> int:
        """
        Get the total number of parameters in the model.
        
        Returns:
            Total parameter count.
        """
        return sum(p.numel() for p in self.parameters())
