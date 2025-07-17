import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Define column names for the dataset
COLUMNS = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

def load_data(filepath):
    """
    Loads the Iris dataset from the given file path.
    """
    df = pd.read_csv(filepath, names=COLUMNS)
    return df

def split_features_labels(df):
    """
    Splits the DataFrame into features (X) and labels (y).
    """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess(filepath):
    """
    Full preprocessing pipeline: Load data, split into X/y, and then train/test.
    Returns X_train, X_test, y_train, y_test
    """
    df = load_data(filepath)
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    return X_train, X_test, y_train, y_test
