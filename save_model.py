import pickle
import os
from model import load_data, preprocess_data, train_model

def save_model():
    """
    Train and save the model, scaler, and feature names as pickle files
    """
    print("Loading data...")
    data = load_data("house_data.csv")  # Will create synthetic data if file not found
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    model_path = "models/house_price_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save the scaler
    scaler_path = "models/feature_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Save the feature names
    features_path = "models/feature_names.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Feature names saved to {features_path}")
    
    return model_path, scaler_path, features_path

if __name__ == "__main__":
    save_model() 