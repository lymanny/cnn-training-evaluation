
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==================================================
# Lesson 06: Evaluation Metrics
# (Accuracy, Precision, Recall, F1-score)
# ==================================================

# Evaluation Metrics:
# Measure how well a model predicts correct labels.
#
# Accuracy = percentage of all correct predictions.
# Precision = how often predicted positives are correct.
# Recall = how many actual positives are detected.
# F1-score = balances Precision and Recall.
#
# Skin lesion example:
# 0 = Benign, 1 = Malignant (positive class).


# ==================================================
# 1. Create Example Data
# ==================================================

# Actual labels = correct answers.
actual = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

# Predicted labels = model's example predictions.
predicted = torch.tensor([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])


# ==================================================
# 2. Calculate Evaluation Metrics
# ==================================================

# .numpy() = convert PyTorch tensors to NumPy arrays.
y_true = actual.numpy()
y_pred = predicted.numpy()

# Accuracy = correct predictions / all predictions.
accuracy = accuracy_score(y_true, y_pred)

# Precision = TP / (TP + FP).
precision = precision_score(y_true, y_pred)

# Recall = TP / (TP + FN).
recall = recall_score(y_true, y_pred)

# F1-score = 2 * Precision * Recall / (Precision + Recall).
f1 = f1_score(y_true, y_pred)


# ==================================================
# 3. Display Results
# ==================================================

# :.2% = display as a percentage with 2 decimals.
print("=== Evaluation Metrics ===")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")


# ==================================================
# 4. Count TP, TN, FP, FN
# ==================================================

# TP = actual 1, predicted 1.
tp = ((actual == 1) & (predicted == 1)).sum().item()

# TN = actual 0, predicted 0.
tn = ((actual == 0) & (predicted == 0)).sum().item()

# FP = actual 0, predicted 1.
fp = ((actual == 0) & (predicted == 1)).sum().item()

# FN = actual 1, predicted 0.
fn = ((actual == 1) & (predicted == 0)).sum().item()

print("\n=== Classification Results ===")
print("True Positive (TP):", tp)
print("True Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
