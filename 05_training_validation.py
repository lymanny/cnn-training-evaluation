
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==================================================
# Lesson 05: Training and Validation
# ==================================================

# Training:
# Updates model weights using training data.
#
# Validation:
# Checks the model using separate data.
# Validation does NOT update model weights.
#
# Training Loss = error on training data.
# Validation Loss = error on validation data.


# ==================================================
# 1. Create Training and Validation Data
# ==================================================

# Create 100 input values.
x = torch.linspace(0, 10, 100).unsqueeze(1)

# Correct answers follow y = 2x + 1.
y = 2 * x + 1

# Split into 80 training and 20 validation samples.
train_dataset = TensorDataset(x[:80], y[:80])
val_dataset = TensorDataset(x[80:], y[80:])

# DataLoader = groups samples into batches.
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


# ==================================================
# 2. Create Model
# ==================================================

# Linear(1, 1) = Fully Connected layer.
model = nn.Linear(1, 1)

# MSELoss = Mean Squared Error.
# Measures squared prediction errors.
criterion = nn.MSELoss()

# Adam = optimizer that updates model weights.
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 20


# ==================================================
# 3. Training and Validation
# ==================================================

for epoch in range(1, epochs + 1):

    # Training mode = enables training behavior.
    model.train()
    train_loss = 0

    for inputs, targets in train_loader:

        # Clear previous gradients.
        optimizer.zero_grad()

        # Forward propagation.
        predictions = model(inputs)

        # Calculate prediction error.
        loss = criterion(predictions, targets)

        # Backpropagation.
        loss.backward()

        # Update model weights.
        optimizer.step()

        # Add batch loss, weighted by batch size.
        train_loss += loss.item() * inputs.size(0)

    # Average training loss per sample.
    train_loss /= len(train_dataset)

    # Evaluation mode = validation behavior.
    model.eval()
    val_loss = 0

    # Disable gradients during validation.
    with torch.no_grad():

        for inputs, targets in val_loader:

            # Predict using validation data.
            predictions = model(inputs)

            # Calculate validation loss.
            loss = criterion(predictions, targets)

            val_loss += loss.item() * inputs.size(0)

    # Average validation loss per sample.
    val_loss /= len(val_dataset)

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )
