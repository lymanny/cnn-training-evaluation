import torch
import torch.nn as nn

# ==================================================
# Lesson 01: Loss Functions (Cross-Entropy / BCE)
# ==================================================

# Loss Function:
# Measures how wrong the model's prediction is.
# Low loss = good prediction.
# High loss = bad prediction.


# ==================================================
# 1. Cross-Entropy Loss
# ==================================================

# Used for multiple classes (e.g., MNIST digits 0-9).

# Create a loss function for multiclass classification.
cross_entropy = nn.CrossEntropyLoss()

# Correct answer is digit 7.
target = torch.tensor([7])

# Create 10 prediction scores for one image.
good = torch.zeros(1, 10)
bad = torch.zeros(1, 10)

# Good: Give the correct digit 7 a high score.
good[0, 7] = 5.0

# Bad: Give the incorrect digit 3 a high score.
bad[0, 3] = 5.0

# Calculate and display prediction errors.
print("=== Cross-Entropy ===")
print("Good loss:", cross_entropy(good, target).item())
print("Bad loss:", cross_entropy(bad, target).item())


# ==================================================
# 2. Binary Cross-Entropy (BCE)
# ==================================================

# Used for binary classification (e.g., Benign / Malignant).

# Create a binary loss function.
bce = nn.BCEWithLogitsLoss()

# Correct answer is 1 (Malignant).
target_binary = torch.tensor([1.0])

# Good: Positive score favors class 1.
good_binary = torch.tensor([3.0])

# Bad: Negative score favors class 0.
bad_binary = torch.tensor([-3.0])

# Calculate and display prediction errors.
print("\n=== Binary Cross-Entropy ===")
print("Good loss:", bce(good_binary, target_binary).item())
print("Bad loss:", bce(bad_binary, target_binary).item())