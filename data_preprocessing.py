# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Standardize column names by making them lowercase and stripping whitespace
    df.columns = df.columns.str.strip().str.lower()

    # Print the standardized column names
    print("Standardized columns in the dataset:", df.columns.tolist())

    # Convert the date column to datetime if it exists
    if 'orderdate' in df.columns:
        df['orderdate'] = pd.to_datetime(df['orderdate'])

    # Drop any rows with missing values
    df.dropna(inplace=True)

    # Extract relevant features for model training
    features = df[['orderitemquantity', 'totalitemquantity']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler
