import os
import numpy as np
import pandas as pd

def generate_bike_data(n_samples = 1500, random_state = 46):
    """To Generate synthetic historical bike-share data for Helsinki."""
    np.random.seed(random_state)
    
    stations = ['Kamppi', 'Rautatientori', 'Kallio', 'Töölö', 'Punavuori']
    
    data = {
        'station': np.random.choice(stations, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'temperature': np.random.normal(12.0, 8.0, n_samples), # Celsius
        'wind_speed': np.random.uniform(1.0, 12.0, n_samples), # m/s
        'precipitation': np.random.choice([0.0, 0.5, 2.5, 5.0], n_samples, p = [0.7, 0.15, 0.1, 0.05]),
        'is_weekend': np.random.choice([0, 1], n_samples, p = [0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Simulate realistic bike rental counts based on features
    # Commuter peaks around 8 AM and 5 PM (17:00), warm weather boosts rentals, rain drops them.
    base_rentals = 20
    
    # Peak hour effect
    hour_effect = np.where((df['hour'].isin([7, 8, 9, 16, 17, 18])), 35, 5)
    
    # Weather effects
    temp_effect = df['temperature'] * 1.5
    rain_effect = df['precipitation'] * -10.0
    weekend_effect = df['is_weekend'] * -15.0 # Fewer commuters on weekends
    
    rentals = (
        base_rentals
        + hour_effect
        + temp_effect
        + rain_effect
        + weekend_effect
        + np.random.normal(0, 5, n_samples)
    )
    
    df['rental_count'] = np.clip(rentals, 0, None).astype(int)
    return df

if __name__ == "__main__":
    print("Generating Helsinki bike rental dataset...")
    df = generate_bike_data()
    
    # Ensure data folder exists
    os.makedirs('data', exist_ok = True)
    file_path = os.path.join('data', 'bike_rentals.csv')
    
    # Save to CSV
    df.to_csv(file_path, index = False)
    print(f"Dataset successfully created and saved to {file_path}!")
    print("\nFirst 5 rows of your dataset:")
    print(df.head())