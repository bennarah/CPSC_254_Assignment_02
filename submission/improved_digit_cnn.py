import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Number of training epochs
NUM_EPOCHS = 12

# Define a CNN model for MNIST digit classification
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # Second convolution layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout layer to help reduce overfitting
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # First convolution block
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Second convolution block
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten before fully connected layers
        x = x.view(-1, 32 * 7 * 7)

        # Apply dropout and fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def main():
    # Convert images to tensors and normalize pixel values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST training and testing datasets
    train_dataset = datasets.MNIST(root='', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='', train=False, transform=transform, download=True)

    # Create batches for training and testing
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Create the model and move it to CPU or GPU
    model = CNN().to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 1, 28, 28)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate accuracy on test dataset
    correct, total = 0, 0
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 1, 28, 28)
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "improved_digit_cnn.pth")
    print("Model saved as improved_digit_cnn.pth")


if __name__ == "__main__":
    main()
