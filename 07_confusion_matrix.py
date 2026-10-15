
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==================================================
# Lesson 07: Confusion Matrix
# ==================================================

# Confusion Matrix:
# Shows correct and incorrect classification results.
#
# TP = True Positive (Malignant predicted Malignant).
# TN = True Negative (Benign predicted Benign).
# FP = False Positive (Benign predicted Malignant).
# FN = False Negative (Malignant predicted Benign).
#
# 0 = Benign
# 1 = Malignant


# ==================================================
# 1. Create Example Data
# ==================================================

# Actual = correct labels.
actual = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

# Predicted = example model predictions.
predicted = torch.tensor([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])


# ==================================================
# 2. Calculate Confusion Matrix
# ==================================================

# .numpy() = convert a tensor into a NumPy array.
y_true = actual.numpy()
y_pred = predicted.numpy()

# confusion_matrix() = counts predictions for each class.
# labels=[0, 1] = keep Benign first, Malignant second.
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("=== Confusion Matrix ===")
print(cm)

# Extract TN, FP, FN, TP from the 2x2 matrix.
tn, fp, fn, tp = cm.ravel()

print("\n=== Classification Results ===")
print("True Positive (TP):", tp)
print("True Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)


# ==================================================
# 3. Visualize Confusion Matrix
# ==================================================

# Display the matrix as a chart.
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)

# plot() = draw the confusion matrix.
display.plot(cmap="Blues", values_format="d")

# Set chart title.
plt.title("Skin Lesion Confusion Matrix")

# Show the chart.
plt.show()
