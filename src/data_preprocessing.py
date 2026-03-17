import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def features_target_split(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


def train_test_split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    return X_train, X_test, y_train, y_test


def feature_scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled,scaler



def preprocess_data(file_path):
    df = load_data(file_path)
    X, y = features_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_scaled, X_test_scaled,scaler = feature_scaling(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    file_path = 'data/creditcard.csv'
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(file_path)
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("Data preprocessing completed.")