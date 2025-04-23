import pandas as pd
import joblib

def predict(data_path, model_path, output_path):
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    
    df['prediction'] = model.predict(df[['elevation']])
    df.to_csv(output_path, index=False)
    print("Prediction saved to", output_path)
