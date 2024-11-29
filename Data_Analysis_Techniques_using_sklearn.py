import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_breast_cancer, fetch_california_housing

# Load datasets
wine = load_wine()
breast_cancer = load_breast_cancer()
california_housing = fetch_california_housing()

# Create pandas DataFrames
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
breast_cancer_df['target'] = breast_cancer.target

california_housing_df = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
california_housing_df['target'] = california_housing.target

# Select only the first 5 features for each dataset
# wine_df = pd.DataFrame(data=wine.data[:, :5], columns=wine.feature_names[:5])
# wine_df['target'] = wine.target

def analyze_dataset(df, name):
    """Analyze and visualize the given dataset."""
    print(f"\n--- {name.upper()} DATASET ---")
    print("Head of the dataset:\n", df.head())
    print("\nTail of the dataset:\n", df.tail())
    print("\nFeature Columns:\n", list(df.columns[:-1]))  # Exclude the target column
    print("\nTarget Column:\n", df.columns[-1])

    # Pairplot for feature visualization
    plt.figure(figsize=(10, 6))
    sns.pairplot(df, vars=df.columns[:-1], hue='target', diag_kind='kde', height=1.5)
    plt.suptitle(f"Pairplot of {name} Dataset", y=1.02)
    plt.show()

    # Histograms for all features
    df.hist(bins=50, figsize=(20, 15))
    plt.suptitle(f"Histograms of {name} Dataset Features", y=1.02)
    plt.show()

    # Correlation Matrix Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation Matrix of {name} Dataset")
    plt.show()


# Analyze the datasets
analyze_dataset(wine_df, "wine")
analyze_dataset(breast_cancer_df, "breast cancer")
analyze_dataset(california_housing_df, "california housing")