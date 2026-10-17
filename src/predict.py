import os
import joblib
import pandas as pd

def predict_demand():
    # 1. Check if the trained model exists
    model_path = os.path.join('data', 'bike_model.pkl')
    if not os.path.exists(model_path):
        print("Trained model not found! Please run train_model.py first.")
        return
        
    # 2. Load the saved model pipeline
    print("Loading trained bike-share model...")
    model = joblib.load(model_path)
    
    # 3. Create sample scenarios we want to predict
    # Let's test a rush hour vs a rainy weekend day
    scenarios = pd.DataFrame([
        {
            'station': 'Central Station',
            'hour': 8,              # Morning rush hour
            'temperature': 20.0,    # Nice warm day
            'wind_speed': 3.0,
            'precipitation': 0.0,   # No rain
            'is_weekend': 0         # Weekday commuter
        },
        {
            'station': 'Central Station',
            'hour': 14,             # Midday off-peak
            'temperature': 5.0,     # Cold day
            'wind_speed': 8.0,
            'precipitation': 2.5,   # Raining heavily
            'is_weekend': 1         # Weekend
        }
    ])
    
    # 4. Make predictions using our trained pipeline
    predictions = model.predict(scenarios)
    
    # 5. Display the results
    print("\n--- Helsinki Bike Demand Predictions ---")
    for i, row in scenarios.iterrows():
        predicted_count = round(predictions[i])
        print(f"\nScenario {i+1}:")
        print(f" - Station: {row['station']}")
        print(f" - Time: Hour {row['hour']} | Temp: {row['temperature']}°C | Rain: {row['precipitation']}mm | Weekend: {row['is_weekend']}")
        print(f" => **Predicted Rental Count: {predicted_count} bikes**")

if __name__ == "__main__":
    predict_demand()