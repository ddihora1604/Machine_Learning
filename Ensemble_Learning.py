import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- WINE DATASET ---
print("\n--- WINE DATASET ---")
# Load Dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implement Bagging
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging.fit(X_train_scaled, y_train)
y_pred_bagging = bagging.predict(X_test_scaled)
bagging_results = {
    'accuracy': accuracy_score(y_test, y_pred_bagging),
    'precision': precision_score(y_test, y_pred_bagging, average='weighted'),
    'recall': recall_score(y_test, y_pred_bagging, average='weighted'),
    'f1': f1_score(y_test, y_pred_bagging, average='weighted')
}

# Implement AdaBoost
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train_scaled, y_train)
y_pred_adaboost = adaboost.predict(X_test_scaled)
adaboost_results = {
    'accuracy': accuracy_score(y_test, y_pred_adaboost),
    'precision': precision_score(y_test, y_pred_adaboost, average='weighted'),
    'recall': recall_score(y_test, y_pred_adaboost, average='weighted'),
    'f1': f1_score(y_test, y_pred_adaboost, average='weighted')
}

# Implement Gradient Boosting
gradboost = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradboost.fit(X_train_scaled, y_train)
y_pred_gradboost = gradboost.predict(X_test_scaled)
gradboost_results = {
    'accuracy': accuracy_score(y_test, y_pred_gradboost),
    'precision': precision_score(y_test, y_pred_gradboost, average='weighted'),
    'recall': recall_score(y_test, y_pred_gradboost, average='weighted'),
    'f1': f1_score(y_test, y_pred_gradboost, average='weighted')
}

# Wine Dataset Results DataFrame
wine_results = pd.DataFrame({
    'Bagging': bagging_results,
    'AdaBoost': adaboost_results,
    'Gradient Boosting': gradboost_results
}).T

# --- Visualizations for Wine Dataset ---
wine_results.plot(kind='bar', figsize=(10, 6), title='Wine Dataset Performance Metrics', color=['#4CAF50', '#FF9800', '#2196F3'])
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Print Wine Dataset Results
print("\nWine Dataset Results:")
print(wine_results)

# --- BREAST CANCER DATASET ---
print("\n--- BREAST CANCER DATASET ---")
# Load Dataset
bc = load_breast_cancer()
X, y = bc.data, bc.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implement Bagging
bagging.fit(X_train_scaled, y_train)
y_pred_bagging = bagging.predict(X_test_scaled)
bagging_results_bc = {
    'accuracy': accuracy_score(y_test, y_pred_bagging),
    'precision': precision_score(y_test, y_pred_bagging),
    'recall': recall_score(y_test, y_pred_bagging),
    'f1': f1_score(y_test, y_pred_bagging)
}

# Implement AdaBoost
adaboost.fit(X_train_scaled, y_train)
y_pred_adaboost = adaboost.predict(X_test_scaled)
adaboost_results_bc = {
    'accuracy': accuracy_score(y_test, y_pred_adaboost),
    'precision': precision_score(y_test, y_pred_adaboost),
    'recall': recall_score(y_test, y_pred_adaboost),
    'f1': f1_score(y_test, y_pred_adaboost)
}

# Implement Gradient Boosting
gradboost.fit(X_train_scaled, y_train)
y_pred_gradboost = gradboost.predict(X_test_scaled)
gradboost_results_bc = {
    'accuracy': accuracy_score(y_test, y_pred_gradboost),
    'precision': precision_score(y_test, y_pred_gradboost),
    'recall': recall_score(y_test, y_pred_gradboost),
    'f1': f1_score(y_test, y_pred_gradboost)
}

# Breast Cancer Dataset Results DataFrame
bc_results = pd.DataFrame({
    'Bagging': bagging_results_bc,
    'AdaBoost': adaboost_results_bc,
    'Gradient Boosting': gradboost_results_bc
}).T

# --- Visualizations for Breast Cancer Dataset ---
bc_results.plot(kind='bar', figsize=(10, 6), title='Breast Cancer Dataset Performance Metrics', color=['#4CAF50', '#FF9800', '#2196F3'])
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Print Breast Cancer Dataset Results
print("\nBreast Cancer Dataset Results:")
print(bc_results)

# --- CALIFORNIA HOUSING DATASET ---
print("\n--- CALIFORNIA HOUSING DATASET ---")
# Load Dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implement Bagging Regressor
bagging_reg = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42)
bagging_reg.fit(X_train_scaled, y_train)
y_pred_bagging = bagging_reg.predict(X_test_scaled)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)

# Implement AdaBoost Regressor
adaboost_reg = AdaBoostRegressor(n_estimators=50, random_state=42)
adaboost_reg.fit(X_train_scaled, y_train)
y_pred_adaboost = adaboost_reg.predict(X_test_scaled)
mse_adaboost = mean_squared_error(y_test, y_pred_adaboost)

# Implement Gradient Boosting Regressor
gradboost_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gradboost_reg.fit(X_train_scaled, y_train)
y_pred_gradboost = gradboost_reg.predict(X_test_scaled)
mse_gradboost = mean_squared_error(y_test, y_pred_gradboost)

# California Housing Dataset Results DataFrame
california_results = pd.DataFrame({
    'Bagging': pd.Series({'MSE': mse_bagging}),
    'AdaBoost': pd.Series({'MSE': mse_adaboost}),
    'Gradient Boosting': pd.Series({'MSE': mse_gradboost})
}).T

# --- Visualizations for California Housing Dataset ---
california_results.plot(kind='bar', figsize=(10, 6), title='California Housing Dataset Performance Metrics (MSE)', color=['#FF5722', '#03A9F4', '#8BC34A'])
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Print California Housing Dataset Results
print("\nCalifornia Housing Dataset Results:")
print(california_results)
