import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

def train_model(data_path, save_model_path):
    df = pd.read_csv(data_path)
    
    X = df[['elevation']]  # Add more features as you go
    y = df['flood_risk']   # 0: No Risk, 1: Risk
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = XGBClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, save_model_path)
    print("Model saved to", save_model_path)
