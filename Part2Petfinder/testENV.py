from torchvision import transforms
from torch.utils.data import DataLoader
from DataLoader import NumericalDataset
from PetfinderNN import get_metadata_model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim

# Create the dataset
train_dataset = NumericalDataset(
    csv_file='Part2Petfinder/petfinder-pawpularity-score/train.csv',
    include_target=True
)

# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model
model = get_metadata_model()
criterion = nn.MSELoss()  # Regression loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for metadata, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(metadata)
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")