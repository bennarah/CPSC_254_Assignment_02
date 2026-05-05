#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load dataset
data_dir = "datasets/animals"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load pretrained ResNet18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace final fully connected layer to match dataset classes
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Basic model information
total_params = sum(p.numel() for p in model.parameters())
conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
batchnorm_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

print("\nModel Information:")
print("Model name: ResNet18")
print(f"Number of classes: {num_classes}")
print(f"Class labels: {dataset.classes}")
print(f"Total parameters: {total_params}")
print(f"Number of convolutional layers: {len(conv_layers)}")
print(f"Number of linear layers: {len(linear_layers)}")
print(f"Number of batch normalization layers: {len(batchnorm_layers)}")
print(f"First conv layer filters: {conv_layers[0].out_channels}")
print("Other architecture components: residual blocks, batch normalization, ReLU, adaptive average pooling, fully connected output layer\n")

# Training loop
num_epochs = 3
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
end_time = time.time()
classification_time = end_time - start_time

print(f"\nClassification Accuracy: {accuracy:.2f}%")
print(f"Total Classification Time: {classification_time:.2f} seconds")

# Save model
torch.save(model.state_dict(), "animal_classifier.pth")
print("Model saved as animal_classifier.pth")