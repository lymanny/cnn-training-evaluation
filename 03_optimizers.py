import torch
import torch.nn as nn
import torch.optim as optim

# ==================================================
# Lesson 03: Optimizers (Adam / SGD)
# ==================================================

# Optimizer:
# Updates model weights to reduce prediction errors.
#
# SGD (Stochastic Gradient Descent):
# Updates weights using gradients and a learning rate.
#
# Adam (Adaptive Moment Estimation):
# Adjusts weight updates using current and past gradients.


# ==================================================
# 1. Create a Simple Model
# ==================================================

# Linear(1, 1) = 1 input and 1 output.
# bias=False = use only one learnable weight.
model = nn.Linear(1, 1, bias=False)

# no_grad() = change the weight without tracking gradients.
# fill_(2.0) = set the starting weight to 2.0.
with torch.no_grad():
    model.weight.fill_(2.0)

# tensor() = create tensors containing numbers.
# x = input (3), target = correct answer (12).
x = torch.tensor([[3.0]])
target = torch.tensor([[12.0]])

# MSELoss = Mean Squared Error.
# Measures squared prediction errors.
criterion = nn.MSELoss()


# ==================================================
# 2. SGD Optimizer
# ==================================================

# SGD = Stochastic Gradient Descent.
# model.parameters() = model's learnable weights.
# lr = learning rate (controls update size).
sgd = optim.SGD(model.parameters(), lr=0.1)

print("=== SGD Optimizer ===")

# epoch = one complete pass through training data.
# range(1, 6) = repeat training 5 times.
for epoch in range(1, 6):

    # Clear gradients from the previous update.
    sgd.zero_grad()

    # Forward propagation: calculate prediction.
    prediction = model(x)

    # Compare prediction with the correct answer.
    loss = criterion(prediction, target)

    # Backpropagation: calculate gradients.
    loss.backward()

    # SGD updates the model weight.
    sgd.step()

    # Display loss and updated weight.
    print(
        f"Epoch {epoch} | "
        f"Loss: {loss.item():.4f} | "
        f"Weight: {model.weight.item():.4f}"
    )


# ==================================================
# 3. Adam Optimizer
# ==================================================

# Reset the weight to 2.0 for a fair comparison.
with torch.no_grad():
    model.weight.fill_(2.0)

# Adam = Adaptive Moment Estimation.
# Uses current and past gradients to adapt updates.
adam = optim.Adam(model.parameters(), lr=0.1)

print("\n=== Adam Optimizer ===")

for epoch in range(1, 6):

    # Clear previous gradients.
    adam.zero_grad()

    # Calculate prediction and loss.
    prediction = model(x)
    loss = criterion(prediction, target)

    # Calculate gradients.
    loss.backward()

    # Adam updates the model weight.
    adam.step()

    # Display loss and updated weight.
    print(
        f"Epoch {epoch} | "
        f"Loss: {loss.item():.4f} | "
        f"Weight: {model.weight.item():.4f}"
    )