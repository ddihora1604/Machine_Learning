import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Wine Dataset - Naive Bayes Classification
# ---------------------------------------------------------
wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# Split Dataset
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Train Naive Bayes Classifier
nb_wine = GaussianNB()
nb_wine.fit(X_train_wine, y_train_wine)

# Evaluate
y_pred_wine = nb_wine.predict(X_test_wine)
results_wine = {
    'Accuracy': accuracy_score(y_test_wine, y_pred_wine),
    'Precision': precision_score(y_test_wine, y_pred_wine, average='weighted'),
    'Recall': recall_score(y_test_wine, y_pred_wine, average='weighted'),
    'F1 Score': f1_score(y_test_wine, y_pred_wine, average='weighted')
}

print("Wine Dataset - Naive Bayes Results:")
print(pd.DataFrame([results_wine], index=['Naive Bayes']))

# Confusion Matrix
cm_wine = confusion_matrix(y_test_wine, y_pred_wine, labels=[0, 1, 2])
sns.heatmap(cm_wine, annot=True, fmt="d", cmap="Blues", xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title("Confusion Matrix - Wine Dataset - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Bar Plot for Performance Metrics - Wine Dataset
metrics_wine = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values_wine = [results_wine['Accuracy'], results_wine['Precision'], results_wine['Recall'], results_wine['F1 Score']]

plt.figure(figsize=(8, 6))
sns.barplot(x=metrics_wine, y=values_wine, palette='viridis')
plt.title("Performance Metrics - Wine Dataset - Naive Bayes")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()

# ---------------------------------------------------------
# Breast Cancer Dataset - Naive Bayes Classification
# ---------------------------------------------------------
bc = load_breast_cancer()
X_bc, y_bc = bc.data, bc.target

# Split Dataset
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

# Train Naive Bayes Classifier
nb_bc = GaussianNB()
nb_bc.fit(X_train_bc, y_train_bc)

# Evaluate
y_pred_bc = nb_bc.predict(X_test_bc)
results_bc = {
    'Accuracy': accuracy_score(y_test_bc, y_pred_bc),
    'Precision': precision_score(y_test_bc, y_pred_bc),
    'Recall': recall_score(y_test_bc, y_pred_bc),
    'F1 Score': f1_score(y_test_bc, y_pred_bc)
}

print("\nBreast Cancer Dataset - Naive Bayes Results:")
print(pd.DataFrame([results_bc], index=['Naive Bayes']))

# Confusion Matrix
cm_bc = confusion_matrix(y_test_bc, y_pred_bc, labels=[0, 1])
sns.heatmap(cm_bc, annot=True, fmt="d", cmap="Blues", xticklabels=bc.target_names, yticklabels=bc.target_names)
plt.title("Confusion Matrix - Breast Cancer Dataset - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Bar Plot for Performance Metrics - Breast Cancer Dataset
metrics_bc = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values_bc = [results_bc['Accuracy'], results_bc['Precision'], results_bc['Recall'], results_bc['F1 Score']]

plt.figure(figsize=(8, 6))
sns.barplot(x=metrics_bc, y=values_bc, palette='viridis')
plt.title("Performance Metrics - Breast Cancer Dataset - Naive Bayes")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()

# ---------------------------------------------------------
# California Housing Dataset - Naive Bayes Classification
# ---------------------------------------------------------
california = fetch_california_housing()
X_california, y_california = california.data, california.target

# Discretize the target variable into bins (classification task)
bins = np.linspace(y_california.min(), y_california.max(), 4)  # Divide into 3 bins
y_california_binned = np.digitize(y_california, bins) - 1  # Convert bins into classes (0, 1, 2)

# Split Dataset
X_train_california, X_test_california, y_train_california, y_test_california = train_test_split(
    X_california, y_california_binned, test_size=0.2, random_state=42)

# Train Naive Bayes Classifier
nb_california = GaussianNB()
nb_california.fit(X_train_california, y_train_california)

# Evaluate
y_pred_california = nb_california.predict(X_test_california)
results_california = {
    'Accuracy': accuracy_score(y_test_california, y_pred_california),
    'Precision': precision_score(y_test_california, y_pred_california, average='weighted'),
    'Recall': recall_score(y_test_california, y_pred_california, average='weighted'),
    'F1 Score': f1_score(y_test_california, y_pred_california, average='weighted')
}

print("\nCalifornia Housing Dataset - Naive Bayes Results:")
print(pd.DataFrame([results_california], index=['Naive Bayes']))

# Confusion Matrix
cm_california = confusion_matrix(y_test_california, y_pred_california, labels=[0, 1, 2])
sns.heatmap(cm_california, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix - California Housing Dataset - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Bar Plot for Performance Metrics - California Housing Dataset
metrics_california = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values_california = [results_california['Accuracy'], results_california['Precision'], results_california['Recall'], results_california['F1 Score']]

plt.figure(figsize=(8, 6))
sns.barplot(x=metrics_california, y=values_california, palette='viridis')
plt.title("Performance Metrics - California Housing Dataset - Naive Bayes")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()
