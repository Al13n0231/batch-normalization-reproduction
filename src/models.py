import torch
import torch.nn as nn
from batch_norm import BatchNorm1d

class SimpleNet(nn.Module):
    """
    Simple 3-layer fully-connected network for MNIST.
    From Section 4.1: "3 fully-connected hidden layers with 100 activations each"
    with sigmoid nonlinearity.
    """
    def __init__(self, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        
        # Architecture: 784 -> 100 -> 100 -> 100 -> 10
        # Note: No bias if using BN (Section 3.2)
        self.fc1 = nn.Linear(784, 100, bias=not use_bn)
        self.fc2 = nn.Linear(100, 100, bias=not use_bn)
        self.fc3 = nn.Linear(100, 100, bias=not use_bn)
        self.fc4 = nn.Linear(100, 10, bias=True)  # Output layer keeps bias
        
        if use_bn:
            self.bn1 = BatchNorm1d(100)
            self.bn2 = BatchNorm1d(100)
            self.bn3 = BatchNorm1d(100)
        
        # Initialize weights with small random Gaussian (Section 4.1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to small random Gaussian values"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, training=True):
        """
        Forward pass.
        
        Args:
            x: Input images (N, 1, 28, 28) or (N, 784)
            training: Whether in training mode (affects BN)
        
        Returns:
            Logits for 10 classes
        """
        # Flatten input
        x = x.view(x.size(0), -1)  # (N, 784)
        
        # Layer 1
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1.forward(x, training)
        x = torch.sigmoid(x)
        
        # Layer 2
        x = self.fc2(x)
        if self.use_bn:
            x = self.bn2.forward(x, training)
        x = torch.sigmoid(x)
        
        # Layer 3
        x = self.fc3(x)
        if self.use_bn:
            x = self.bn3.forward(x, training)
        x = torch.sigmoid(x)
        
        # Output layer (no BN, no activation)
        x = self.fc4(x)
        
        return x
    
    def get_bn_parameters(self):
        """Get all BatchNorm parameters for optimizer"""
        if not self.use_bn:
            return []
        params = []
        params.extend(self.bn1.parameters())
        params.extend(self.bn2.parameters())
        params.extend(self.bn3.parameters())
        return params
    
    def __repr__(self):
        return f"SimpleNet(use_bn={self.use_bn})"