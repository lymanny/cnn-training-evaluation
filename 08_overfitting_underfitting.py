
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==================================================
# Lesson 08: Overfitting and Underfitting
# ==================================================

# Underfitting:
# Model fails to learn useful patterns.
# Both training and validation loss remain high.
#
# Overfitting:
# Model fits training data well but performs poorly
# on new validation data.
#
# Good Fit:
# Model learns patterns that generalize to new data.


# ==================================================
# 1. Create Example Data
# ==================================================

# manual_seed() = makes random results reproducible.
torch.manual_seed(42)

# linspace() = creates evenly spaced input values.
x = torch.linspace(-3, 3, 200).unsqueeze(1)

# Create noisy data following a curved pattern.
y = torch.sin(3 * x) + 0.2 * torch.randn_like(x)

# Choose 15 random samples for training.
indices = torch.randperm(200)
train_ids = indices[:15]
val_ids = indices[15:]

x_train, y_train = x[train_ids], y[train_ids]
x_val, y_val = x[val_ids], y[val_ids]


# ==================================================
# 2. Create Neural Network
# ==================================================

# Sequential() = connects layers in order.
# Linear() = Fully Connected (FC) layer.
# Tanh() = activation function that adds nonlinearity.
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

# MSELoss() = calculates squared prediction errors.
criterion = nn.MSELoss()

# Adam() = optimizer that updates model weights.
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ==================================================
# 3. Train and Validate
# ==================================================

epochs = 500
train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):

    # Training mode.
    model.train()

    # Clear previous gradients.
    optimizer.zero_grad()

    # Forward propagation.
    predictions = model(x_train)

    # Calculate training loss.
    train_loss = criterion(predictions, y_train)

    # Backpropagation: calculate gradients.
    train_loss.backward()

    # Update model weights.
    optimizer.step()

    # Evaluation mode.
    model.eval()

    # Validation does not update model weights.
    with torch.no_grad():
        val_predictions = model(x_val)
        val_loss = criterion(val_predictions, y_val)

    # Save losses for plotting.
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train: {train_loss.item():.4f} | "
            f"Val: {val_loss.item():.4f}"
        )


# ==================================================
# 4. Visualize Training and Validation Loss
# ==================================================

# Plot loss values across training epochs.
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.title("Overfitting and Underfitting")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
