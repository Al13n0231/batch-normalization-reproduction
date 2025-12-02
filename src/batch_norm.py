import torch

class BatchNorm1d:
    """
    Batch Normalization for fully-connected layers (1D).
    Implements Algorithm 1 and Algorithm 2 from Ioffe & Szegedy 2015.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Args:
            num_features: Number of features (C from an expected input of size (N, C))
            eps: Small constant for numerical stability (epsilon in paper)
            momentum: Momentum for running mean/var updates (typically 0.1)
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters (γ and β)
        self.gamma = torch.ones(num_features, requires_grad=True)
        self.beta = torch.zeros(num_features, requires_grad=True)
        
        # Running statistics (for inference)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long)
        
    def forward(self, x, training=True):
        """
        Forward pass implementing Algorithm 1.
        
        Args:
            x: Input tensor of shape (N, C) where N is batch size, C is num_features
            training: Boolean indicating training vs inference mode
            
        Returns:
            y: Normalized, scaled, and shifted output
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input (got {x.dim()}D input)")
        
        batch_size, num_features = x.shape
        
        if num_features != self.num_features:
            raise ValueError(f"Expected input with {self.num_features} features (got {num_features})")
        
        if training:
            # TRAINING MODE: Use batch statistics (Algorithm 1)
            
            # Step 1: Compute mini-batch mean (μ_B in paper)
            batch_mean = x.mean(dim=0)  # Shape: (C,)
            
            # Step 2: Compute mini-batch variance (σ²_B in paper)
            batch_var = x.var(dim=0, unbiased=False)  # Shape: (C,)
            
            # Step 3: Normalize (x̂ in paper)
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            
            # Step 4: Scale and shift (y in paper)
            y = self.gamma * x_normalized + self.beta
            
            # Update running statistics for inference (Algorithm 2, line 10)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                
                # Bessel's correction: multiply by m/(m-1)
                unbiased_var = batch_var * batch_size / (batch_size - 1)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_var
                
                self.num_batches_tracked += 1
            
            return y
            
        else:
            # INFERENCE MODE: Use running statistics (Algorithm 2, line 11)
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            y = self.gamma * x_normalized + self.beta
            return y
    
    def parameters(self):
        """Return learnable parameters (for optimizer)"""
        return [self.gamma, self.beta]
    
    def __repr__(self):
        return f"BatchNorm1d({self.num_features}, eps={self.eps}, momentum={self.momentum})"
    

class BatchNorm2d:
    """
    Batch Normalization for convolutional layers (2D).
    Implements Section 3.2 of the paper.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Args:
            num_features: Number of channels (C from expected input of size (N, C, H, W))
            eps: Small constant for numerical stability
            momentum: Momentum for running mean/var updates
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters (one γ and β per channel)
        self.gamma = torch.ones(num_features, requires_grad=True)
        self.beta = torch.zeros(num_features, requires_grad=True)
        
        # Running statistics
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long)
    
    def forward(self, x, training=True):
        """
        Forward pass for convolutional BN (Section 3.2).
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            training: Boolean indicating training vs inference mode
            
        Returns:
            y: Normalized output
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input)")
        
        batch_size, num_channels, height, width = x.shape
        
        if num_channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels (got {num_channels})")
        
        if training:
            # TRAINING MODE: Compute statistics over (N, H, W) for each channel
            
            # Compute mean over (batch, height, width) dimensions
            # Shape: (C,)
            batch_mean = x.mean(dim=(0, 2, 3))
            
            # Compute variance over (batch, height, width) dimensions
            # Shape: (C,)
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # Normalize: Reshape for broadcasting
            # batch_mean and batch_var shape: (C,) → (1, C, 1, 1)
            x_normalized = (x - batch_mean.view(1, -1, 1, 1)) / \
                          torch.sqrt(batch_var.view(1, -1, 1, 1) + self.eps)
            
            # Scale and shift: gamma and beta shape (C,) → (1, C, 1, 1)
            y = self.gamma.view(1, -1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1)
            
            # Update running statistics
            with torch.no_grad():
                # Effective batch size for Conv2d is N * H * W
                effective_batch_size = batch_size * height * width
                
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                   self.momentum * batch_mean
                
                # Bessel's correction
                unbiased_var = batch_var * effective_batch_size / (effective_batch_size - 1)
                self.running_var = (1 - self.momentum) * self.running_var + \
                                  self.momentum * unbiased_var
                
                self.num_batches_tracked += 1
            
            return y
            
        else:
            # INFERENCE MODE: Use running statistics
            
            x_normalized = (x - self.running_mean.view(1, -1, 1, 1)) / \
                          torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)
            
            y = self.gamma.view(1, -1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1)
            
            return y
    
    def parameters(self):
        """Return learnable parameters (for optimizer)"""
        return [self.gamma, self.beta]
    
    def __repr__(self):
        return f"BatchNorm2d({self.num_features}, eps={self.eps}, momentum={self.momentum})"