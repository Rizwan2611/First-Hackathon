# genai_model_training.py

import tensorflow as tf
from keras import layers, models
from data_preprocessing import load_data
import joblib

def build_genai_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_shape[0])  # Output shape matches input
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def train_genai_model(file_path):
    data, scaler = load_data(file_path)
    model = build_genai_model((data.shape[1],))

    # Train the model
    model.fit(data, data, epochs=50, batch_size=8, validation_split=0.2)

    # Save the trained model and scaler
    model.save('../models/trained_genai_model.h5')
    joblib.dump(scaler, '../models/scaler.pkl')
    print("Model trained and saved.")

if __name__ == "__main__":
    train_genai_model('../data/ML-Dataset.csv')
