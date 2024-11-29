from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_wine, load_breast_cancer, fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# --- WINE DATASET ---
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

X_wine = wine_df.drop('target', axis=1)
y_wine = wine_df['target']

# Split data into training and testing sets
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model_wine = LinearRegression()
model_wine.fit(X_train_wine, y_train_wine)

# Make predictions on the test set
y_pred_wine = model_wine.predict(X_test_wine)

# Evaluate the model
mse_wine = mean_squared_error(y_test_wine, y_pred_wine)
r2_wine = r2_score(y_test_wine, y_pred_wine)

print("\n--- WINE DATASET ---")
print(f"Mean Squared Error: {mse_wine}")
print(f"R-squared: {r2_wine}")

# Plot: Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_wine, y_pred_wine, alpha=0.6)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs. Predicted Values (Wine Dataset)")
plt.plot([min(y_test_wine), max(y_test_wine)], [min(y_test_wine), max(y_test_wine)], color='red', linestyle='--')
plt.show()


# --- BREAST CANCER DATASET ---
breast_cancer = load_breast_cancer()
breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
breast_cancer_df['target'] = breast_cancer.target

X_breast_cancer = breast_cancer_df.drop('target', axis=1)
y_breast_cancer = breast_cancer_df['target']

# Split data into training and testing sets
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model_bc = LinearRegression()
model_bc.fit(X_train_bc, y_train_bc)

# Make predictions on the test set
y_pred_bc = model_bc.predict(X_test_bc)

# Evaluate the model
mse_bc = mean_squared_error(y_test_bc, y_pred_bc)
r2_bc = r2_score(y_test_bc, y_pred_bc)

print("\n--- BREAST CANCER DATASET ---")
print(f"Mean Squared Error: {mse_bc}")
print(f"R-squared: {r2_bc}")

# Plot: Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_bc, y_pred_bc, alpha=0.6)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs. Predicted Values (Breast Cancer Dataset)")
plt.plot([min(y_test_bc), max(y_test_bc)], [min(y_test_bc), max(y_test_bc)], color='red', linestyle='--')
plt.show()


# --- CALIFORNIA HOUSING DATASET ---
california_housing = fetch_california_housing()
california_housing_df = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
california_housing_df['target'] = california_housing.target

X_california = california_housing_df.drop('target', axis=1)
y_california = california_housing_df['target']

# Split data into training and testing sets
X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(X_california, y_california, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model_cal = LinearRegression()
model_cal.fit(X_train_cal, y_train_cal)

# Make predictions on the test set
y_pred_cal = model_cal.predict(X_test_cal)

# Evaluate the model
mse_cal = mean_squared_error(y_test_cal, y_pred_cal)
r2_cal = r2_score(y_test_cal, y_pred_cal)

print("\n--- CALIFORNIA HOUSING DATASET ---")
print(f"Mean Squared Error: {mse_cal}")
print(f"R-squared: {r2_cal}")

# Plot: Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_cal, y_pred_cal, alpha=0.6)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs. Predicted Values (California Housing Dataset)")
plt.plot([min(y_test_cal), max(y_test_cal)], [min(y_test_cal), max(y_test_cal)], color='red', linestyle='--')
plt.show()
