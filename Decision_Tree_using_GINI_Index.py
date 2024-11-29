import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

# --- WINE DATASET ---
print("\n--- WINE DATASET ---")
# Load Dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier with Gini Index
dt_wine = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_wine.fit(X_train, y_train)

# Evaluate
y_pred_wine = dt_wine.predict(X_test)
results_wine = {
    'Accuracy': accuracy_score(y_test, y_pred_wine),
    'Precision': precision_score(y_test, y_pred_wine, average='weighted'),
    'Recall': recall_score(y_test, y_pred_wine, average='weighted'),
    'F1 Score': f1_score(y_test, y_pred_wine, average='weighted')
}

print(pd.DataFrame([results_wine], index=['Wine Dataset']))

# Confusion Matrix
cm_wine = confusion_matrix(y_test, y_pred_wine)
sns.heatmap(cm_wine, annot=True, fmt="d", cmap="Blues", xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title("Confusion Matrix - Wine Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Additional Plot 1: Decision Tree Structure for Wine Dataset ---
plt.figure(figsize=(12, 8))
plot_tree(dt_wine, filled=True, feature_names=wine.feature_names, class_names=wine.target_names, rounded=True)
plt.title("Decision Tree - Wine Dataset")
plt.show()

# --- Additional Plot 2: Learning Curve for Wine Dataset ---
train_sizes, train_scores, test_scores = learning_curve(dt_wine, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score", color="green")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Cross-validation Score", color="blue")
plt.title("Learning Curve - Wine Dataset")
plt.xlabel("Training Set Size")
plt.ylabel("Score")
plt.legend()
plt.show()

# --- BREAST CANCER DATASET ---
print("\n--- BREAST CANCER DATASET ---")
# Load Dataset
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier with Gini Index
dt_bc = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_bc.fit(X_train, y_train)

# Evaluate
y_pred_bc = dt_bc.predict(X_test)
results_bc = {
    'Accuracy': accuracy_score(y_test, y_pred_bc),
    'Precision': precision_score(y_test, y_pred_bc, average='weighted'),
    'Recall': recall_score(y_test, y_pred_bc, average='weighted'),
    'F1 Score': f1_score(y_test, y_pred_bc, average='weighted')
}

print(pd.DataFrame([results_bc], index=['Breast Cancer Dataset']))

# Confusion Matrix
cm_bc = confusion_matrix(y_test, y_pred_bc)
sns.heatmap(cm_bc, annot=True, fmt="d", cmap="Blues", xticklabels=breast_cancer.target_names, yticklabels=breast_cancer.target_names)
plt.title("Confusion Matrix - Breast Cancer Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Additional Plot 1: Decision Tree Structure for Breast Cancer Dataset ---
plt.figure(figsize=(12, 8))
plot_tree(dt_bc, filled=True, feature_names=breast_cancer.feature_names, class_names=breast_cancer.target_names, rounded=True)
plt.title("Decision Tree - Breast Cancer Dataset")
plt.show()

# --- Additional Plot 2: Learning Curve for Breast Cancer Dataset ---
train_sizes, train_scores, test_scores = learning_curve(dt_bc, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score", color="green")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Cross-validation Score", color="blue")
plt.title("Learning Curve - Breast Cancer Dataset")
plt.xlabel("Training Set Size")
plt.ylabel("Score")
plt.legend()
plt.show()










import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Convert the target to binary classification (above or below median)
y_binary = (y > np.median(y)).astype(int)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Train Decision Tree Classifier with GINI index
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)

# Make predictions
y_pred = dt_gini.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Below Median", "Above Median"], yticklabels=["Below Median", "Above Median"])
plt.title("Confusion Matrix - Decision Tree (GINI Index)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_gini, filled=True, feature_names=data.feature_names, class_names=["Below Median", "Above Median"], rounded=True)
plt.title("Decision Tree - California Housing Dataset")
plt.show()

# Print Decision Tree Rules
tree_rules = export_text(dt_gini, feature_names=data.feature_names)
print("\nDecision Tree Rules:\n")
print(tree_rules)
