import sys
sys.path.insert(0, '../src')
from models import SimpleNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def quick_test(use_bn, lr, steps=5000):
    """Quick test with different learning rates"""
    device = torch.device('cpu')
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = SimpleNet(use_bn=use_bn).to(device)
    params = list(model.parameters())
    if use_bn:
        params.extend(model.get_bn_parameters())
    
    optimizer = optim.SGD(params, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    train_iter = iter(train_loader)
    
    print(f"Testing: BN={use_bn}, LR={lr}")
    
    for step in tqdm(range(steps)):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)
        
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images, training=True)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check for explosion
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ❌ Training collapsed! Loss = {loss.item()}")
            return None
    
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, training=False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"  ✅ Final Accuracy: {accuracy:.2%}\n")
    return accuracy

if __name__ == "__main__":
    print("="*60)
    print("Testing Section 3.3: Higher Learning Rates with BN")
    print("="*60)
    
    results = {}
    
    # Baseline: lr=0.01
    print("\n1. Baseline LR = 0.01:")
    results['without_bn_lr001'] = quick_test(use_bn=False, lr=0.01)
    results['with_bn_lr001'] = quick_test(use_bn=True, lr=0.01)
    
    # 5× higher: lr=0.05
    print("\n2. Higher LR = 0.05 (5×):")
    results['without_bn_lr005'] = quick_test(use_bn=False, lr=0.05)
    results['with_bn_lr005'] = quick_test(use_bn=True, lr=0.05)
    
    # 10× higher: lr=0.1
    print("\n3. Very High LR = 0.1 (10×):")
    results['without_bn_lr01'] = quick_test(use_bn=False, lr=0.1)
    results['with_bn_lr01'] = quick_test(use_bn=True, lr=0.1)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"LR=0.01: Without BN = {results.get('without_bn_lr001', 'N/A')}, With BN = {results.get('with_bn_lr001', 'N/A')}")
    print(f"LR=0.05: Without BN = {results.get('without_bn_lr005', 'N/A')}, With BN = {results.get('with_bn_lr005', 'N/A')}")
    print(f"LR=0.1:  Without BN = {results.get('without_bn_lr01', 'N/A')}, With BN = {results.get('with_bn_lr01', 'N/A')}")