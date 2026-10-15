
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==================================================
# Lesson 04: Learning Rate, Batch Size, and Epochs
# ==================================================

# Hyperparameters:
# Settings we choose to control model training.
#
# Learning Rate:
# Controls the size of weight updates.
#
# Batch Size:
# Number of samples processed in one training step.
#
# Epoch:
# One complete pass through the training dataset.


# ==================================================
# 1. Set Hyperparameters
# ==================================================

learning_rate = 0.01  # Controls weight update size
batch_size = 20       # Process 20 samples per batch
epochs = 5            # Train over the dataset 5 times


# ==================================================
# 2. Create Training Data
# ==================================================

# Create 100 input values.
x = torch.linspace(0, 10, 100).unsqueeze(1)

# Correct answers follow y = 2x + 1.
y = 2 * x + 1

# TensorDataset = groups inputs and correct answers.
dataset = TensorDataset(x, y)

# DataLoader = divides the dataset into batches.
# shuffle=True = randomizes sample order each epoch.
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# ==================================================
# 3. Create Model
# ==================================================

# Linear(1, 1) = FC layer with 1 input and 1 output.
model = nn.Linear(1, 1)

# MSELoss = measures squared prediction errors.
criterion = nn.MSELoss()

# SGD = optimizer that updates model weights.
# lr = learning rate.
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate
)


# ==================================================
# 4. Train Model
# ==================================================

# One epoch = process all training samples once.
for epoch in range(1, epochs + 1):

    total_loss = 0

    # Process one batch at a time.
    for inputs, targets in loader:

        # Clear old gradients.
        optimizer.zero_grad()

        # Forward propagation: make predictions.
        predictions = model(inputs)

        # Calculate prediction error.
        loss = criterion(predictions, targets)

        # Backpropagation: calculate gradients.
        loss.backward()

        # Optimizer: update weights.
        optimizer.step()

        # Accumulate loss for this batch.
        total_loss += loss.item()

    # Calculate average loss across batches.
    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch}/{epochs} | "
        f"Loss: {average_loss:.4f}"
    )


# ==================================================
# 5. Display Results
# ==================================================

print("\n=== Training Summary ===")
print("Learning Rate:", learning_rate)
print("Batch Size:", batch_size)
print("Epochs:", epochs)
print("Training Samples:", len(dataset))
print("Batches per Epoch:", len(loader))
print("Total Training Steps:", len(loader) * epochs)
