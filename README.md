# 🚚 Amazon Delivery Time Prediction

This project predicts the delivery time for Amazon orders using geospatial and operational features like store/drop location, agent rating, and order time. It includes a Streamlit web app and tracks model performance with MLflow.

## 📌 Features
- Predict delivery time using:
  - Distance between store and drop location
  - Agent rating
  - Order time (hour)
- Supports 4 regression models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Interactive Streamlit web app
- MLflow for experiment tracking

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Geopy for distance calculation
- Joblib for model saving
- Streamlit for UI
- MLflow for model tracking

## 🚀 How to Run
1. Install dependencies:  
   `pip install -r requirements.txt`
2. Train the model and log with MLflow.
3. Run the Streamlit app:  
   `streamlit run app.py`
4. Access MLflow UI:  
   `mlflow ui`

## 📊 Results
- Best model: **Gradient Boosting Regressor**
- Metrics:
  - RMSE: ~42.23
  - MAE: ~31.65
  - R²: ~0.32

## ✅ Future Enhancements
- Add traffic/weather data
- Deploy on cloud (e.g. Streamlit Cloud, AWS)
- Enable real-time input via APIs

   
