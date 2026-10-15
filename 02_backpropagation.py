
import torch

# ==================================================
# Lesson 02: Backpropagation
# ==================================================

# Backpropagation:
# Calculates gradients to help reduce prediction errors.

# Weight = a number the model learns.
# requires_grad=True = track calculations for gradients.
weight = torch.tensor(2.0, requires_grad=True)

# Input and correct answer.
x = torch.tensor(3.0)
target = torch.tensor(12.0)

# Learning rate = controls the size of weight updates.
learning_rate = 0.1

# ==================================================
# 1. Forward Propagation
# ==================================================

# Prediction = input multiplied by weight.
prediction = weight * x

# Loss = squared difference from the correct answer.
loss = (prediction - target) ** 2

print("=== Before Backpropagation ===")
print("Weight:", weight.item())
print("Prediction:", prediction.item())
print("Loss:", loss.item())


# ==================================================
# 2. Backpropagation
# ==================================================

# backward() = calculate the gradient of the loss.
loss.backward()

# .grad = gradient stored for the weight.
print("\n=== Backpropagation ===")
print("Gradient:", weight.grad.item())


# ==================================================
# 3. Update Weight
# ==================================================

# no_grad() = do not track the manual weight update.
with torch.no_grad():

    # Update weight to help reduce loss.
    weight -= learning_rate * weight.grad

# Clear the old gradient.
weight.grad.zero_()

# Calculate prediction and loss again.
new_prediction = weight * x
new_loss = (new_prediction - target) ** 2

print("\n=== After Weight Update ===")
print("New weight:", weight.item())
print("New prediction:", new_prediction.item())
print("New loss:", new_loss.item())
