# incremental_learning.py
import pandas as pd
import tensorflow as tf
from data_preprocessing import load_data
import joblib

def load_trained_model():
    model = tf.keras.models.load_model('../models/trained_genai_model.h5')
    scaler = joblib.load('../models/scaler.pkl')
    return model, scaler

def incremental_learning(new_data_file):
    # Load existing model
    model, scaler = load_trained_model()

    # Load and scale new data
    new_data, _ = load_data(new_data_file)
    new_data_scaled = scaler.transform(new_data)

    # Retrain model with new data
    model.fit(new_data_scaled, new_data_scaled, epochs=10, batch_size=8)
    model.save('../models/trained_genai_model.h5')
    print("Model updated with new data.")

if __name__ == "__main__":
    incremental_learning('../data/ML-Dataset.csv')
