# Helsinki Smart Bike-Share Demand Predictor 🚴‍♂️

An end-to-end Machine Learning project designed to predict public bike-share demand in Helsinki based on weather conditions, temporal features, and spatial factors. This repository demonstrates a production-style machine learning pipeline including vectorized synthetic data generation, preprocessing, model training, evaluation, persistent serialization, and real-time scenario inference.

## 🔬 Project Architecture & Features

```
          +-----------------------+
          |  data_pipeline.py     |  <-- Generates realistic Helsinki bike data
          +-----------+-----------+      using Pandas, NumPy & domain logic
                      |
                      v
          +-----------------------+
          |  data/bike_rentals.csv|  <-- Saved dataset (1,500 samples)
          +-----------+-----------+
                      |
                      v
          +-----------------------+
          |  train_model.py       |  <-- Trains Scikit-Learn Pipeline
          +-----------+-----------+      (StandardScaler + OneHotEncoder + RandomForest)
                      |
                      v
          +-----------------------+
          |  data/bike_model.pkl  |  <-- Model serialization via Joblib
          +-----------+-----------+
                      |
                      v
          +-----------------------+
          |     predict.py        |  <-- Loads model to run scenario-based
          +-----------------------+      demand predictions
```

## 📈 Key Results & Model Performance

The model uses a Scikit-Learn **`Pipeline`** combining feature scaling, categorical encoding, and a **`RandomForestRegressor`** (100 estimators).

* **R² Score:** `0.87` *(Captures ~87% of overall variance in demand)*
* **RMSE (Root Mean Squared Error):** `7.52` rentals *(Average error margin of ~7–8 bikes)*

### Scenario Predictions (Inference Test)

| Scenario | Station | Hour | Temp (°C) | Rain (mm) | Weekend? | **Predicted Rentals** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rush Hour Commute** | Central Station | 08:00 | 20.0°C | 0.0 mm | No | **90 bikes** |
| **Rainy Off-Peak** | Central Station | 14:00 | 5.0°C | 2.5 mm | Yes | **9 bikes** |

---

## 🛠️ Project Structure

```
helsinki-bike-predictor/
├── data/
│   ├── bike_rentals.csv    # Generated dataset
│   └── bike_model.pkl      # Serialized Scikit-Learn pipeline
├── src/
│   ├── data_pipeline.py    # Synthetic data generation engine
│   ├── train_model.py      # Model training & evaluation script
│   └── predict.py          # Real-time inference script
├── .gitignore
└── README.md               # Project documentation
```

---

## 🧠 Core Engineering Principles Applied

* **Vectorization:** Fully vectorized data generation using NumPy (`np.where`, `np.random.normal`) and Pandas to ensure high memory and execution efficiency without iterative loops.
* **Domain Physics & Behavior Modeling:** Encoded realistic urban commuting behavior, such as rush hour spikes (07–09 & 16–18), temperature scaling, precipitation penalties, and weekend commuting drop-offs.
* **Leakage-Safe Preprocessing:** Leveraged Scikit-Learn `ColumnTransformer` and `Pipeline` structures to apply `StandardScaler` to numerical inputs and `OneHotEncoder` to spatial categories without data leakage.
* **Model Persistence:** Integrated `Joblib` for efficient binary serialization of the fitted machine learning pipeline.

---

## 🚀 How to Run the Project

### 1. Prerequisites & Environment Setup
Clone the repository and install required Python packages:
```bash
git clone [https://github.com/nnshaikhml/helsinki-bike-predictor.git](https://github.com/nnshaikhml/helsinki-bike-predictor.git)
cd helsinki-bike-predictor
pip install pandas numpy scikit-learn joblib
```

### 2. Generate the Dataset
Run the synthetic data pipeline script to construct `data/bike_rentals.csv`:
```bash
python src/data_pipeline.py
```

### 3. Train the Model
Execute the training script to evaluate performance and produce the saved model artifact (`data/bike_model.pkl`):
```bash
python src/train_model.py
```

### 4. Run Scenario Inference
Test predictions across different operational scenarios:
```bash
python src/predict.py
```