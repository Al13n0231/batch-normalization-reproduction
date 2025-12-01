import sys
sys.path.insert(0, '../src')
from cnn_model import SimpleCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def train_cifar10_cnn(use_bn=False, num_steps=50000, batch_size=128, learning_rate=0.01):
    """
    Train CNN on CIFAR-10 with optional Batch Normalization.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training {'WITH' if use_bn else 'WITHOUT'} Batch Normalization")
    
    # CIFAR-10 data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Create model
    model = SimpleCNN(use_bn=use_bn).to(device)
    print(f"Model: {model}")
    
    # Optimizer
    params = list(model.parameters())
    if use_bn:
        params.extend(model.get_bn_parameters())
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    train_iter = iter(train_loader)
    
    train_losses = []
    test_accuracies = []
    steps_recorded = []
    
    print(f"\nTraining for {num_steps} steps...")
    pbar = tqdm(range(num_steps))
    
    for step in pbar:
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
        
        train_losses.append(loss.item())
        
        if step % 1000 == 0:
            test_acc = evaluate(model, test_loader, device)
            test_accuracies.append(test_acc)
            steps_recorded.append(step)
            pbar.set_description(f"Loss: {loss.item():.4f}, Test Acc: {test_acc:.2%}")
    
    final_acc = evaluate(model, test_loader, device)
    test_accuracies.append(final_acc)
    steps_recorded.append(num_steps)
    
    print(f"\nFinal Test Accuracy: {final_acc:.2%}")
    
    return {
        'steps': steps_recorded,
        'test_accuracies': test_accuracies,
        'train_losses': train_losses,
        'final_accuracy': final_acc
    }

def evaluate(model, test_loader, device):
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
    
    model.train()
    return correct / total

def plot_results(results_without, results_with):
    os.makedirs('../results/plots', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(results_without['steps'], 
             [acc * 100 for acc in results_without['test_accuracies']], 
             label='Without BN', marker='o', linewidth=2)
    ax1.plot(results_with['steps'], 
             [acc * 100 for acc in results_with['test_accuracies']], 
             label='With BN', marker='s', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('CIFAR-10 CNN: Test Accuracy vs Training Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Smoothed loss
    window = 100
    def smooth(data, w):
        return [sum(data[max(0, i-w):i+1]) / len(data[max(0, i-w):i+1]) for i in range(len(data))]
    
    ax2.plot(smooth(results_without['train_losses'], window), label='Without BN', alpha=0.7, linewidth=2)
    ax2.plot(smooth(results_with['train_losses'], window), label='With BN', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Training Loss (smoothed)')
    ax2.set_title('Training Loss Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = '../results/plots/cifar10_cnn_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("CIFAR-10 CNN Experiment - Section 3.2 Reproduction")
    print("="*60)
    
    TEST_MODE = False  # Set False for full training
    num_steps = 5000 if TEST_MODE else 50000
    
    if TEST_MODE:
        print("\n⚠️  TEST MODE (5000 steps, ~20 min)")
        print("Set TEST_MODE = False for full training (~3-4 hours)\n")
    
    # Train without BN
    print("\n" + "="*60)
    print("Training CNN WITHOUT Batch Normalization")
    print("="*60)
    results_without = train_cifar10_cnn(use_bn=False, num_steps=num_steps)
    
    # Train with BN
    print("\n" + "="*60)
    print("Training CNN WITH Batch Normalization")
    print("="*60)
    results_with = train_cifar10_cnn(use_bn=True, num_steps=num_steps)
    
    # Compare
    print("\n" + "="*60)
    print("CIFAR-10 CNN RESULTS")
    print("="*60)
    print(f"Without BN: {results_without['final_accuracy']:.2%}")
    print(f"With BN:    {results_with['final_accuracy']:.2%}")
    print(f"Improvement: {(results_with['final_accuracy'] - results_without['final_accuracy']) * 100:.2f} percentage points")
    
    plot_results(results_without, results_with)