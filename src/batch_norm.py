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