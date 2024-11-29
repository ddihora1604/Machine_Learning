!pip install scikit-learn-extra
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------
# Wine Dataset
# --------------------------------------------------------

# Load Dataset
wine_data = load_wine()
X_wine, y_wine = wine_data.data, wine_data.target

# Train K-Medoids (3 clusters, as there are 3 classes in the wine dataset)
kmedoids_wine = KMedoids(n_clusters=3, random_state=42)
kmedoids_wine.fit(X_wine)

# Predictions and evaluation
y_pred_wine = kmedoids_wine.labels_
silhouette_wine = silhouette_score(X_wine, y_pred_wine)

print("Wine Dataset - K-Medoids Clustering Results:")
print(f"Silhouette Score: {silhouette_wine:.3f}")

# Plotting the clusters (for 2D projection)
plt.figure(figsize=(8, 6))
plt.scatter(X_wine[:, 0], X_wine[:, 1], c=y_pred_wine, cmap='viridis', alpha=0.7)
plt.title("K-Medoids Clustering on Wine Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# --------------------------------------------------------
# Breast Cancer Dataset
# --------------------------------------------------------

# Load Dataset
bc_data = load_breast_cancer()
X_bc, y_bc = bc_data.data, bc_data.target

# Standardize the data
scaler_bc = StandardScaler()
X_bc_scaled = scaler_bc.fit_transform(X_bc)

# Train K-Medoids (2 clusters, as it's a binary classification problem)
kmedoids_bc = KMedoids(n_clusters=2, random_state=42)
kmedoids_bc.fit(X_bc_scaled)

# Predictions and evaluation
y_pred_bc = kmedoids_bc.labels_
silhouette_bc = silhouette_score(X_bc_scaled, y_pred_bc)

print("\nBreast Cancer Dataset - K-Medoids Clustering Results:")
print(f"Silhouette Score: {silhouette_bc:.3f}")

# Plotting the clusters (for 2D projection of the first two features)
plt.figure(figsize=(8, 6))
plt.scatter(X_bc_scaled[:, 0], X_bc_scaled[:, 1], c=y_pred_bc, cmap='viridis', alpha=0.7)
plt.title("K-Medoids Clustering on Breast Cancer Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# --------------------------------------------------------
# California Housing Dataset
# --------------------------------------------------------

# Load Dataset
california_data = fetch_california_housing()
X_california, y_california = california_data.data, california_data.target

# Standardize the data
scaler_california = StandardScaler()
X_california_scaled = scaler_california.fit_transform(X_california)

# Train K-Medoids (2 clusters, as we are classifying above or below median of housing prices)
kmedoids_california = KMedoids(n_clusters=2, random_state=42)
kmedoids_california.fit(X_california_scaled)

# Predictions and evaluation
y_pred_california = kmedoids_california.labels_
silhouette_california = silhouette_score(X_california_scaled, y_pred_california)

print("\nCalifornia Housing Dataset - K-Medoids Clustering Results:")
print(f"Silhouette Score: {silhouette_california:.3f}")

# Plotting the clusters (for 2D projection of the first two features)
plt.figure(figsize=(8, 6))
plt.scatter(X_california_scaled[:, 0], X_california_scaled[:, 1], c=y_pred_california, cmap='viridis', alpha=0.7)
plt.title("K-Medoids Clustering on California Housing Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
