import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class SimpleNN(nn.Module):
    """A simple feedforward neural network for image classification."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_data():
    """Load and preprocess the MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """Train the neural network model."""
    losses = []
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                losses.append(loss.item())
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    return losses

def evaluate_model(model, test_loader):
    """Evaluate the model on test data."""
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy

def plot_training(losses):
    """Plot training loss over batches."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Batch (x100)')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()

def main():
    # Hyperparameters
    input_size = 784  # 28x28 pixels
    hidden_size = 128
    num_classes = 10   # Digits 0-9
    learning_rate = 0.001
    num_epochs = 5
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Initialize model, loss, and optimizer
    model = SimpleNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("Starting training...")
    losses = train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # Evaluate the model
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, test_loader)
    
    # Plot training loss
    plot_training(losses)
    print("Training complete! Check 'training_loss.png' for the loss curve.")

if __name__ == "__main__":
    main()
