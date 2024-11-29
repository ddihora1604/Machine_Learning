import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------
# Wine Dataset
# --------------------------------------------------------
wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# Split Dataset
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Scale Data
scaler_wine = StandardScaler()
X_train_wine_scaled = scaler_wine.fit_transform(X_train_wine)
X_test_wine_scaled = scaler_wine.transform(X_test_wine)

# Initialize Models for Wine Dataset
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}

# Train and Evaluate Models for Wine Dataset
results_wine = {}
for name, model in models.items():
    print(f"Training {name} for Wine Dataset...")
    model.fit(X_train_wine_scaled, y_train_wine)
    y_pred_wine = model.predict(X_test_wine_scaled)
    results_wine[name] = {
        'accuracy': accuracy_score(y_test_wine, y_pred_wine),
        'precision': precision_score(y_test_wine, y_pred_wine, average='weighted'),
        'recall': recall_score(y_test_wine, y_pred_wine, average='weighted'),
        'f1': f1_score(y_test_wine, y_pred_wine, average='weighted')
    }
    print(f"{name} performance:", results_wine[name])

# Convert the results to DataFrame
df_results_wine = pd.DataFrame(results_wine).T
print("\nWine Dataset Performance Results:")
print(df_results_wine)

# Heatmap Visualization for Wine Dataset
plt.figure(figsize=(8, 6))
sns.heatmap(df_results_wine, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('Performance Comparison of Models for Wine Dataset')
plt.tight_layout()
plt.show()

# Bar Chart Visualization for Wine Dataset
metrics = ['accuracy', 'precision', 'recall', 'f1']
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Performance Comparison of Models for Wine Dataset (Bar Charts)', fontsize=16)

for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    df_results_wine[metric].plot(kind='bar', ax=ax)
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_ylabel(metric.capitalize())
    ax.set_xticklabels(df_results_wine.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Breast Cancer Dataset
# --------------------------------------------------------
bc = load_breast_cancer()
X_bc, y_bc = bc.data, bc.target

# Split Dataset
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

# Scale Data
scaler_bc = StandardScaler()
X_train_bc_scaled = scaler_bc.fit_transform(X_train_bc)
X_test_bc_scaled = scaler_bc.transform(X_test_bc)

# Train and Evaluate Models for Breast Cancer Dataset
results_bc = {}
for name, model in models.items():
    print(f"Training {name} for Breast Cancer Dataset...")
    model.fit(X_train_bc_scaled, y_train_bc)
    y_pred_bc = model.predict(X_test_bc_scaled)
    results_bc[name] = {
        'accuracy': accuracy_score(y_test_bc, y_pred_bc),
        'precision': precision_score(y_test_bc, y_pred_bc),
        'recall': recall_score(y_test_bc, y_pred_bc),
        'f1': f1_score(y_test_bc, y_pred_bc)
    }
    print(f"{name} performance:", results_bc[name])

# Convert the results to DataFrame
df_results_bc = pd.DataFrame(results_bc).T
print("\nBreast Cancer Dataset Performance Results:")
print(df_results_bc)

# Heatmap Visualization for Breast Cancer Dataset
plt.figure(figsize=(8, 6))
sns.heatmap(df_results_bc, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('Performance Comparison of Models for Breast Cancer Dataset')
plt.tight_layout()
plt.show()

# Bar Chart Visualization for Breast Cancer Dataset
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Performance Comparison of Models for Breast Cancer Dataset (Bar Charts)', fontsize=16)

for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    df_results_bc[metric].plot(kind='bar', ax=ax)
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_ylabel(metric.capitalize())
    ax.set_xticklabels(df_results_bc.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# California Housing Dataset
# --------------------------------------------------------
california = fetch_california_housing()
X_california, y_california = california.data, california.target

# For classification, convert target to binary (above or below median)
y_california_bin = (y_california > np.median(y_california)).astype(int)

# Split Dataset
X_train_california, X_test_california, y_train_california, y_test_california = train_test_split(X_california, y_california_bin, test_size=0.2, random_state=42)

# Scale Data
scaler_california = StandardScaler()
X_train_california_scaled = scaler_california.fit_transform(X_train_california)
X_test_california_scaled = scaler_california.transform(X_test_california)

# Train and Evaluate Models for California Housing Dataset
results_california = {}
for name, model in models.items():
    print(f"Training {name} for California Housing Dataset...")
    model.fit(X_train_california_scaled, y_train_california)
    y_pred_california = model.predict(X_test_california_scaled)
    results_california[name] = {
        'accuracy': accuracy_score(y_test_california, y_pred_california),
        'precision': precision_score(y_test_california, y_pred_california),
        'recall': recall_score(y_test_california, y_pred_california),
        'f1': f1_score(y_test_california, y_pred_california)
    }
    print(f"{name} performance:", results_california[name])

# Convert the results to DataFrame
df_results_california = pd.DataFrame(results_california).T
print("\nCalifornia Housing Dataset Performance Results:")
print(df_results_california)

# Heatmap Visualization for California Housing Dataset
plt.figure(figsize=(8, 6))
sns.heatmap(df_results_california, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('Performance Comparison of Models for California Housing Dataset')
plt.tight_layout()
plt.show()

# Bar Chart Visualization for California Housing Dataset
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Performance Comparison of Models for California Housing Dataset (Bar Charts)', fontsize=16)

for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    df_results_california[metric].plot(kind='bar', ax=ax)
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_ylabel(metric.capitalize())
    ax.set_xticklabels(df_results_california.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()
