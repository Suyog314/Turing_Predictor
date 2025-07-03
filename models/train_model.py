
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.turing_dataset import TuringPatternDataset
from models.cnn_model import TuringCNN

# === Hyperparameters ===
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# === Dataset ===
dataset = TuringPatternDataset(csv_file='metadata.csv', image_dir='patterns')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model, Loss, Optimizer ===
model = TuringCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for images, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss / len(dataloader):.6f}")

# === Save Model ===
torch.save(model.state_dict(), 'results/turing_cnn.pth')
print("✅ Model saved to results/turing_cnn.pth")
