from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_clean_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    return df

def explore_load_data():
    df = load_clean_data()
    
    print("First five rows in dataset:")
    print(df.head(), "\n")

    print("Dataframe dimensions:")
    print(df.shape, "\n")

    print("Data types and non-null counts:")
    print(df.info(), "\n")

    print("Descriptive statistics:")
    print(df.describe(), "\n")

    print("Class distribution (target variable):")
    print(df['target'].value_counts(), "\n")

    print("Missing values in each column:")
    print(df.isnull().sum(), "\n")

if __name__ == "__main__":
    explore_load_data()