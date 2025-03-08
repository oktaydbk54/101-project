import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Function to load and prepare data
def load_data(filepath):
    """
    Load housing data from a CSV file
    If you don't have a dataset, this function provides a sample synthetic dataset
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded from {filepath}")
    except:
        print("Creating synthetic data for demonstration")
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: square footage, bedrooms, bathrooms, age of house, lot size
        sqft = np.random.normal(2000, 500, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples) + np.random.random(n_samples)
        age = np.random.randint(0, 50, n_samples)
        lot_size = np.random.normal(8000, 2000, n_samples)
        
        # Target: house price (with some noise)
        price = 100000 + 150 * sqft + 15000 * bedrooms + 25000 * bathrooms - 1000 * age + 2 * lot_size
        price = price + np.random.normal(0, 50000, n_samples)  # Add noise
        
        # Create DataFrame
        data = pd.DataFrame({
            'sqft': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'lot_size': lot_size,
            'price': price
        })
    
    return data

# Function to preprocess data
def preprocess_data(data):
    """
    Preprocess the data: handle missing values, scale features, etc.
    """
    # Handle missing values (if any)
    data = data.dropna()
    
    # Define features and target
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# Function to train the model
def train_model(X_train, y_train):
    """
    Train a linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model and print performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: ${mse:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Print coefficients
    print("\nFeature Coefficients:")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"{feature}: ${coef:.2f}")
    print(f"Intercept: ${model.intercept_:.2f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    print("Plot saved as 'prediction_results.png'")
    
    return rmse, r2

# Function to make new predictions
def predict_house_price(model, scaler, features):
    """
    Predict the price of a house based on its features
    
    Args:
        model: Trained linear regression model
        scaler: Fitted StandardScaler
        features: Dictionary of feature values (sqft, bedrooms, etc.)
    
    Returns:
        Predicted price
    """
    # Convert features to DataFrame
    features_df = pd.DataFrame([features])
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    predicted_price = model.predict(features_scaled)[0]
    
    return predicted_price

# Main function
def main():
    # Load data
    data = load_data("house_data.csv")  # Replace with your dataset path or use synthetic data
    
    # Display data summary
    print("\nData Summary:")
    print(data.describe())
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, feature_names)
    
    # Example prediction
    example_house = {
        'sqft': 2500,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'age': 10,
        'lot_size': 9000
    }
    
    predicted_price = predict_house_price(model, scaler, example_house)
    print(f"\nExample Prediction:")
    print(f"House features: {example_house}")
    print(f"Predicted price: ${predicted_price:.2f}")

if __name__ == "__main__":
    main() 