# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

# Load dataset
cancer = load_breast_cancer()

# Convert data into DataFrame
df = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

# Add target column
df["target"] = cancer.target

# Display dataset information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["target"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score
)
# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
# THRESHOLD TUNING

from sklearn.metrics import confusion_matrix

threshold = 0.3

y_pred_new = (y_prob >= threshold).astype(int)

print("\nTHRESHOLD TUNING")
print("Threshold:", threshold)

print("\nNew Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_new))

print("New Precision:",
      precision_score(y_test, y_pred_new))

print("New Recall:",
      recall_score(y_test, y_pred_new))

from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

# Confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()

# SIGMOID FUNCTION

import numpy as np

z = np.linspace(-10, 10, 100)

sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(7, 5))
plt.plot(z, sigmoid)

plt.axhline(y=0.5, linestyle="--")
plt.axvline(x=0, linestyle="--")

plt.xlabel("Input (z)")
plt.ylabel("Probability")
plt.title("Sigmoid Function")
plt.grid()
plt.savefig("sigmoid_function.png")
plt.show()