import torch
import torch.nn as nn
from batch_norm import BatchNorm2d

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 with optional Batch Normalization.
    Demonstrates Section 3.2 of the paper.
    """
    def __init__(self, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        
        # Convolutional layers
        # Input: 3×32×32 (CIFAR-10)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=not use_bn)   # 32×32×32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=not use_bn)  # 16×16×64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not use_bn) # 8×8×128
        
        if use_bn:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)
            self.bn3 = BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 128, bias=not use_bn)  # After 3 poolings: 32→16→8→4
        self.fc2 = nn.Linear(128, 10, bias=True)
        
        if use_bn:
            from batch_norm import BatchNorm1d
            self.bn_fc = BatchNorm1d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Initialize
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, training=True):
        """
        Forward pass.
        
        Args:
            x: Input images (N, 3, 32, 32)
            training: Training mode flag
        
        Returns:
            Logits for 10 classes
        """
        # Conv block 1: 3×32×32 → 32×32×32 → 32×16×16
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1.forward(x, training)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Conv block 2: 32×16×16 → 64×16×16 → 64×8×8
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2.forward(x, training)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Conv block 3: 64×8×8 → 128×8×8 → 128×4×4
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3.forward(x, training)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Flatten: 128×4×4 → 2048
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn_fc.forward(x, training)
        x = torch.relu(x)
        
        x = self.fc2(x)
        
        return x
    
    def get_bn_parameters(self):
        """Get all BatchNorm parameters for optimizer"""
        if not self.use_bn:
            return []
        params = []
        params.extend(self.bn1.parameters())
        params.extend(self.bn2.parameters())
        params.extend(self.bn3.parameters())
        params.extend(self.bn_fc.parameters())
        return params
    
    def __repr__(self):
        return f"SimpleCNN(use_bn={self.use_bn})"