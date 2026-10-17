import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

def train_model():
    # 1. Load the dataset
    data_path = os.path.join('data', 'bike_rentals.csv')
    if not os.path.exists(data_path):
        print("Dataset not found! Please run data_pipeline.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Separate Features (X) and Target (y)
    X = df.drop(columns=['rental_count'])
    y = df['rental_count']
    
    # 3. Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 46)
    
    # 4. Define preprocessing for numeric and categorical features
    numeric_features = ['hour', 'temperature', 'wind_speed', 'precipitation', 'is_weekend']
    categorical_features = ['station']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 5. Create a Machine Learning Pipeline (Preprocessing + Random Forest Regressor)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state = 46))
    ])
    
    # 6. Train the model
    print("Training the Random Forest Regressor model...")
    pipeline.fit(X_train, y_train)
    
    # 7. Evaluate the model
    predictions = pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\nModel Evaluation:")
    print(f" - RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f" - R² Score: {r2:.2f} (closer to 1.0 is better)")
    
    # 8. Save the trained model to disk
    os.makedirs('data', exist_ok=True)
    model_path = os.path.join('data', 'bike_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"\nTrained model successfully saved to {model_path}!")

if __name__ == "__main__":
    train_model()