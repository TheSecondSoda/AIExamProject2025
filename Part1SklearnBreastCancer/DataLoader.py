from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_clean_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    return df

def apply_PCA_to_data(X, y, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    #Convert to dataframe for convenience
    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=X_pca, columns=columns)
    pca_df['target'] = y

    return pca_df

def plot_PCA_2D(pca_df):
    if 'PC1' in pca_df.columns and 'PC2' in pca_df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=pca_df,
            x='PC1', y='PC2',
            hue='target',
            palette=['#ff7f0e', '#1f77b4'],
            alpha=0.7
        )
        plt.title(f"PCA (2D)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("PCA has fewer than 2 components — can't plot in 2D.")

def explore_load_data(df):  
    print("First five rows in dataset:")
    print(df.head(), "\n")

    print("Dataframe dimensions:")
    print(df.shape, "\n")

    print("Data types and non-null counts:")
    print(df.info(), "\n")

    print("Descriptive statistics:")
    print(df.describe(), "\n")

    if 'target' in df.columns:
        print("Class distribution (target variable):")
        print(df['target'].value_counts(), "\n")

    print("Missing values in each column:")
    print(df.isnull().sum(), "\n")

if __name__ == "__main__":
    data = load_clean_data()
    #explore_load_data(data)
    pca_df = apply_PCA_to_data(data.drop(columns=['target']), data['target'], 2)
    explore_load_data(pca_df)
    plot_PCA_2D(pca_df)