import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import SimpleNet

def train_mnist(use_bn=False, num_steps=50000, batch_size=60, learning_rate=0.01):
    """
    Train SimpleNet on MNIST following Section 4.1 of the paper.
    
    Args:
        use_bn: Whether to use Batch Normalization
        num_steps: Total training steps (paper uses 50,000)
        batch_size: Mini-batch size (paper uses 60)
        learning_rate: Learning rate for SGD
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training {'WITH' if use_bn else 'WITHOUT'} Batch Normalization")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Create model
    model = SimpleNet(use_bn=use_bn).to(device)
    print(f"Model: {model}")
    
    # Setup optimizer - include BN parameters if using BN
    params = list(model.parameters())
    if use_bn:
        params.extend(model.get_bn_parameters())
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    train_iter = iter(train_loader)
    
    train_losses = []
    test_accuracies = []
    steps_recorded = []
    
    print(f"\nTraining for {num_steps} steps...")
    pbar = tqdm(range(num_steps))
    
    for step in pbar:
        # Get batch (restart dataloader if exhausted)
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)
        
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images, training=True)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluate every 1000 steps
        if step % 1000 == 0:
            test_acc = evaluate(model, test_loader, device)
            test_accuracies.append(test_acc)
            steps_recorded.append(step)
            pbar.set_description(f"Loss: {loss.item():.4f}, Test Acc: {test_acc:.2%}")
    
    # Final evaluation
    final_acc = evaluate(model, test_loader, device)
    test_accuracies.append(final_acc)
    steps_recorded.append(num_steps)
    
    print(f"\nFinal Test Accuracy: {final_acc:.2%}")
    
    return {
        'steps': steps_recorded,
        'test_accuracies': test_accuracies,
        'train_losses': train_losses,
        'final_accuracy': final_acc,
        'model': model
    }

def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, training=False)  # Inference mode
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    model.train()
    return correct / total

def plot_results(results_without_bn, results_with_bn, save_path='../results/plots/mnist_comparison.png'):
    """Plot comparison of training with and without BN"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot test accuracy
    ax1.plot(results_without_bn['steps'], 
             [acc * 100 for acc in results_without_bn['test_accuracies']], 
             label='Without BN', marker='o', linewidth=2)
    ax1.plot(results_with_bn['steps'], 
             [acc * 100 for acc in results_with_bn['test_accuracies']], 
             label='With BN', marker='s', linewidth=2)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('MNIST Test Accuracy vs Training Steps', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot training loss (smoothed)
    window = 100
    def smooth(data, window_size):
        return [sum(data[max(0, i-window_size):i+1]) / len(data[max(0, i-window_size):i+1]) 
                for i in range(len(data))]
    
    smoothed_without = smooth(results_without_bn['train_losses'], window)
    smoothed_with = smooth(results_with_bn['train_losses'], window)
    
    ax2.plot(smoothed_without, label='Without BN', alpha=0.7, linewidth=2)
    ax2.plot(smoothed_with, label='With BN', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Training Loss (smoothed)', fontsize=12)
    ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("MNIST Experiment - Reproducing Section 4.1")
    print("="*60)
    
    # Train without BN
    print("\n" + "="*60)
    print("Training WITHOUT Batch Normalization")
    print("="*60)
    results_without_bn = train_mnist(use_bn=False, num_steps=50000, batch_size=60)
    
    # Train with BN
    print("\n" + "="*60)
    print("Training WITH Batch Normalization")
    print("="*60)
    results_with_bn = train_mnist(use_bn=True, num_steps=50000, batch_size=60)
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"Without BN - Final Accuracy: {results_without_bn['final_accuracy']:.2%}")
    print(f"With BN    - Final Accuracy: {results_with_bn['final_accuracy']:.2%}")
    print(f"Improvement: {(results_with_bn['final_accuracy'] - results_without_bn['final_accuracy']) * 100:.2f} percentage points")
    
    # Plot results
    plot_results(results_without_bn, results_with_bn)