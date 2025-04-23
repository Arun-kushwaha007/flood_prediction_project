import geopandas as gpd
import rasterio
import pandas as pd
from rasterio.plot import show
import os

def read_elevation_data(raster_path):
    with rasterio.open(raster_path) as src:
        elevation = src.read(1)
        return elevation, src

def extract_features(elevation_data, meta):
    # Example: flatten the elevation array and create a DataFrame
    flat = elevation_data.flatten()
    df = pd.DataFrame({'elevation': flat})
    return df
