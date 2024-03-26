# making the changes for version 1 in the feature-1 branch
# the base code of class assignment q2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Load FashionMNIST dataset
train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet101 model
resnet = models.resnet101(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)  # FashionMNIST has 10 classes

# Move model to device
resnet = resnet.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizers
optimizers = {
    'Adam': optim.Adam(resnet.parameters(), lr=0.001),
    'Adagrad': optim.Adagrad(resnet.parameters(), lr=0.01),
    'RMSprop': optim.RMSprop(resnet.parameters(), lr=0.001)
}

# Training function
def train(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct / total
    return train_loss, train_accuracy
# Define the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
def train_loop(model, optimizer, scheduler, criterion, train_loader, num_epochs=10):
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, optimizer, criterion, train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        scheduler.step()  # Step the scheduler
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    return train_losses, train_accuracies

# Train with each optimizer
results = {}
for optimizer_name, optimizer in optimizers.items():
    print(f"\nTraining with {optimizer_name} optimizer...")
    resnet.fc = nn.Linear(num_ftrs, 10)  # Reset classifier for fair comparison
    resnet = resnet.to(device)
    train_losses, train_accuracies = train_loop(resnet, optimizer, criterion, train_loader)
    results[optimizer_name] = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }

# Plot training curves
plt.figure(figsize=(10, 5))
for optimizer_name, result in results.items():
    plt.plot(result['train_losses'], label=f'{optimizer_name} Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for optimizer_name, result in results.items():
    plt.plot(result['train_accuracies'], label=f'{optimizer_name} Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy Curves')
plt.legend()
plt.show()

# Evaluate model on test set
def test(model, test_loader):
    model.eval()
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.topk(5, 1, True, True)
            predicted = predicted.t()
            correct = predicted.eq(labels.view(1, -1).expand_as(predicted))
            correct_top5 += correct[:5].view(-1).float().sum(0).item()
            total += labels.size(0)

    top5_accuracy = correct_top5 / total
    return top5_accuracy

resnet.eval()
top5_accuracy = test(resnet, test_loader)
print(f"Top-5 Test Accuracy: {top5_accuracy:.4f}")
