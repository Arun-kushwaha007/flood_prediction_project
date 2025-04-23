from scripts.preprocess import read_elevation_data, extract_features
from scripts.train_model import train_model
from scripts.predict_flood import predict

# Step 1: Preprocess
elevation, meta = read_elevation_data("data/raw/odisha_dem.tif")
df = extract_features(elevation, meta)
df['flood_risk'] = (df['elevation'] < 5).astype(int)  # Labeling (example)
df.to_csv("data/processed/train.csv", index=False)

# Step 2: Train
train_model("data/processed/train.csv", "models/flood_model.pkl")

# Step 3: Predict
predict("data/processed/train.csv", "models/flood_model.pkl", "outputs/predictions.csv")
