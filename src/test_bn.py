import torch
from batch_norm import BatchNorm1d

def test_batch_norm():
    """Test BatchNorm1d implementation"""
    print("Testing BatchNorm1d implementation...")
    
    # Test 1: Basic forward pass (training mode)
    print("\n1. Testing training mode...")
    bn = BatchNorm1d(num_features=5)
    x = torch.randn(32, 5)  # Batch size 32, 5 features
    
    y = bn.forward(x, training=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Output mean: {y.mean(dim=0)}")  # Should be close to 0
    print(f"   Output var: {y.var(dim=0, unbiased=False)}")  # Should be close to 1
    
    # Test 2: Inference mode
    print("\n2. Testing inference mode...")
    y_inference = bn.forward(x, training=False)
    print(f"   Output shape: {y_inference.shape}")
    print(f"   Running mean: {bn.running_mean}")
    print(f"   Running var: {bn.running_var}")
    
    # Test 3: Gradient flow
    print("\n3. Testing gradient flow...")
    x_grad = torch.randn(32, 5, requires_grad=True)
    y_grad = bn.forward(x_grad, training=True)
    loss = y_grad.sum()
    loss.backward()
    print(f"   Gamma gradient exists: {bn.gamma.grad is not None}")
    print(f"   Beta gradient exists: {bn.beta.grad is not None}")
    print(f"   Input gradient exists: {x_grad.grad is not None}")
    
    # Test 4: Compare with PyTorch's BatchNorm
    print("\n4. Comparing with PyTorch BatchNorm1d...")
    torch_bn = torch.nn.BatchNorm1d(5, momentum=0.1)
    torch_bn.train()
    
    # Use same input
    x_test = torch.randn(32, 5)
    
    # Our implementation
    our_output = bn.forward(x_test, training=True)
    
    # PyTorch implementation
    torch_output = torch_bn(x_test)
    
    print(f"   Our output mean: {our_output.mean(dim=0)}")
    print(f"   PyTorch output mean: {torch_output.mean(dim=0)}")
    print(f"   Difference: {torch.abs(our_output - torch_output).max().item():.6f}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_batch_norm()