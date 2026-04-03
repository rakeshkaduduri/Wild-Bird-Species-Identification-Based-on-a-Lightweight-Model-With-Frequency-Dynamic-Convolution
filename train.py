import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import BirdCNN

# ----------------------------
# Create models folder
# ----------------------------
os.makedirs("models", exist_ok=True)

# ----------------------------
# Image transformations
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ----------------------------
# Load dataset
# ----------------------------
dataset = datasets.ImageFolder("spectrograms", transform=transform)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ----------------------------
# Class information
# ----------------------------
num_classes = len(dataset.classes)

print("Classes:", dataset.classes)
print("Number of classes:", num_classes)

# ----------------------------
# Model
# ----------------------------
model = BirdCNN(num_classes)

# ----------------------------
# Loss and optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(epochs):

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{epochs}  Loss: {running_loss:.4f}  Accuracy: {accuracy:.2f}%")

# ----------------------------
# Save trained model
# ----------------------------
torch.save(model.state_dict(), "models/bird_model.pth")

print("\nTraining completed!")
print("Model saved as models/bird_model.pth")