import pandas as pd

def load_heart_data(filepath):
    # Load the dataset for heart disease
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target
    return X, y

def load_diabetes_data(filepath):
    # Load the dataset for diabetes
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target
    return X, y

def load_parkinsons_data(filepath):
    # Load the dataset for Parkinson's disease
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target
    return X, y

# Example usage:
heart_data_path = r'C:\Users\sudee\Desktop\Multiple-Disease-Prediction-Model-Deployment-using-StreamLit-main\dataFiles\heart_disease_data.csv'
diabetes_data_path = r'C:\Users\sudee\Desktop\Multiple-Disease-Prediction-Model-Deployment-using-StreamLit-main\dataFiles\diabetes.csv'
parkinsons_data_path = r'C:\Users\sudee\Desktop\Multiple-Disease-Prediction-Model-Deployment-using-StreamLit-main\dataFiles\parkinsons.csv'

X_heart, y_heart = load_heart_data(heart_data_path)
X_diabetes, y_diabetes = load_diabetes_data(diabetes_data_path)
X_parkinsons, y_parkinsons = load_parkinsons_data(parkinsons_data_path)
