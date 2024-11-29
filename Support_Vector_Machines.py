import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot decision boundary
def plot_svm_decision_boundary(X, y, model, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# --------------------------------------------------------
# Wine Dataset
# --------------------------------------------------------

# Load Dataset
wine_data = load_wine()
X_wine, y_wine = wine_data.data, wine_data.target

# Standardize the data
scaler_wine = StandardScaler()
X_wine_scaled = scaler_wine.fit_transform(X_wine)

# Reduce to 2D for visualization
pca_wine = PCA(n_components=2)
X_wine_2d = pca_wine.fit_transform(X_wine_scaled)

# Split Dataset
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine_2d, y_wine, test_size=0.2, random_state=42)

# Train SVM Classifier (Linear Kernel)
svm_wine_linear = SVC(kernel='linear')
svm_wine_linear.fit(X_train_wine, y_train_wine)
plot_svm_decision_boundary(X_wine_2d, y_wine, svm_wine_linear, "Wine Dataset - Linear Kernel")

# Train SVM Classifier (RBF Kernel)
svm_wine_rbf = SVC(kernel='rbf', gamma='auto')
svm_wine_rbf.fit(X_train_wine, y_train_wine)
plot_svm_decision_boundary(X_wine_2d, y_wine, svm_wine_rbf, "Wine Dataset - RBF Kernel")

# --------------------------------------------------------
# Breast Cancer Dataset
# --------------------------------------------------------

# Load Dataset
bc_data = load_breast_cancer()
X_bc, y_bc = bc_data.data, bc_data.target

# Standardize the data
scaler_bc = StandardScaler()
X_bc_scaled = scaler_bc.fit_transform(X_bc)

# Reduce to 2D for visualization
pca_bc = PCA(n_components=2)
X_bc_2d = pca_bc.fit_transform(X_bc_scaled)

# Split Dataset
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc_2d, y_bc, test_size=0.2, random_state=42)

# Train SVM Classifier (Linear Kernel)
svm_bc_linear = SVC(kernel='linear')
svm_bc_linear.fit(X_train_bc, y_train_bc)
plot_svm_decision_boundary(X_bc_2d, y_bc, svm_bc_linear, "Breast Cancer Dataset - Linear Kernel")

# Train SVM Classifier (RBF Kernel)
svm_bc_rbf = SVC(kernel='rbf', gamma='auto')
svm_bc_rbf.fit(X_train_bc, y_train_bc)
plot_svm_decision_boundary(X_bc_2d, y_bc, svm_bc_rbf, "Breast Cancer Dataset - RBF Kernel")

# --------------------------------------------------------
# California Housing Dataset
# --------------------------------------------------------

# Load Dataset
california_data = fetch_california_housing()
X_california, y_california = california_data.data[:, :5], california_data.target  # Limit to 5 features

# Convert target to binary (above or below median)
y_california_bin = (y_california > np.median(y_california)).astype(int)

# Standardize the data
scaler_california = StandardScaler()
X_california_scaled = scaler_california.fit_transform(X_california)

# Reduce to 2D for visualization
pca_california = PCA(n_components=2)
X_california_2d = pca_california.fit_transform(X_california_scaled)

# Split Dataset
X_train_california, X_test_california, y_train_california, y_test_california = train_test_split(X_california_2d, y_california_bin, test_size=0.2, random_state=42)

# Train SVM Classifier (Linear Kernel)
svm_california_linear = SVC(kernel='linear')
svm_california_linear.fit(X_train_california, y_train_california)
plot_svm_decision_boundary(X_california_2d, y_california_bin, svm_california_linear, "California Housing Dataset - Linear Kernel")

# Train SVM Classifier (RBF Kernel)
svm_california_rbf = SVC(kernel='rbf', gamma='auto')
svm_california_rbf.fit(X_train_california, y_train_california)
plot_svm_decision_boundary(X_california_2d, y_california_bin, svm_california_rbf, "California Housing Dataset - RBF Kernel")
