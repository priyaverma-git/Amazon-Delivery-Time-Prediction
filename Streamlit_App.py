import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

# --- Load Models ---
def load_model(model_name):
    model_path = f'models/{model_name}_model.pkl'
    model = joblib.load(model_path)
    return model

# --- Page 1: Home ---
def home_page():
    st.title("\U0001F4E6 Amazon Delivery Time Predictor")
    st.write(
        """
        Welcome to the Amazon Delivery Time Prediction app! This app allows you to predict the delivery time 
        based on various factors such as store location, delivery address, agent rating, and order hour.
        
        **Available Models:**
        - Gradient Boosting Regressor
        - Random Forest Regressor
        - Linear Regression
        - XGBoost Regressor
        
        Navigate to the **Prediction** page to make your first prediction.
        """
    )

def prediction_page():
    st.title("\U0001F4E6 Predict Delivery Time")
    st.subheader("Enter Delivery Details:")

    # Model selection
    model_name = st.selectbox("Select Regression Model", ["GradientBoosting", "RandomForest", "LinearRegression", "XGBoost"])

    # Load model and preprocessor once model is selected
    model = load_model(model_name)
    preprocessor = joblib.load("models/preprocessor.pkl")  # load preprocessor

    with st.form("prediction_form"):
        store_lat = st.number_input("Store Latitude", value=12.9716, format="%.6f")
        store_long = st.number_input("Store Longitude", value=77.5946, format="%.6f")
        drop_lat = st.number_input("Drop Latitude", value=13.0358, format="%.6f")
        drop_long = st.number_input("Drop Longitude", value=77.5970, format="%.6f")
        agent_rating = st.slider("Agent Rating", 1.0, 5.0, 3.0)
        order_hour = st.slider("Order Hour (0 to 23)", min_value=0, max_value=23, value=12)

        submitted = st.form_submit_button("Predict Delivery Time")

    if submitted:
        # Calculate geodesic distance
        distance = geodesic((store_lat, store_long), (drop_lat, drop_long)).km

        # Fixed example area value (used if 'Area' was a feature during training)
        area_value = 'Area'  
        # Prepare input data (adjust columns to match your training set)
        input_data = pd.DataFrame({
            'Distance': [distance],
            'Agent_Rating': [agent_rating],
            'Order_Hour': [order_hour],
            'Area': [area_value]  # Only if 'Area' was used
        })

        # Preprocess input data
        input_processed = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(input_processed)

        st.success(f"\U0001F69A Estimated Delivery Time: **{prediction[0]:.2f} hours**")

# --- Page 3: Model Details ---
def model_details_page():
    st.title("\U0001F4CA Model Details")
    st.write(
        """
        This app offers the following regression models to predict delivery times:

        - **Gradient Boosting Regressor**: An ensemble learning method that builds a model in stages, where each stage corrects the errors of the previous one. It generally works well for complex data.
        
        - **Random Forest Regressor**: Another ensemble method that builds multiple decision trees and averages their predictions. It is robust to overfitting and works well with large datasets.
        
        - **Linear Regression**: A simple, interpretable model that assumes a linear relationship between the input features and the output. It's fast and easy to understand but may underperform on complex data.

        - **XGBoost Regressor**: A scalable and accurate implementation of gradient boosting that offers regularization to reduce overfitting and high performance with large datasets.

        Below are some performance metrics commonly used to evaluate regression models:

        - **Root Mean Squared Error (RMSE)**
        - **Mean Absolute Error (MAE)**
        - **R-squared (R²)**
        """
    )
    st.write("You can experiment with the models on the **Prediction** page.")

# --- Page 4: About ---
def about_page():
    st.title("👨‍💼 About This Project")
    st.write(
        """
        This project is part of an Amazon Delivery Time Prediction initiative. The aim of the project is 
        to accurately predict delivery times for packages based on several features such as:
        
        - Store Location (Latitude and Longitude)
        - Delivery Address (Latitude and Longitude)
        - Agent Rating
        - Time of Order Placement
        
        The model was trained on a dataset with real-world delivery information and was evaluated based on 
        several performance metrics.
        
        **Technologies Used:**
        - Python, Streamlit
        - Machine Learning (Gradient Boosting, Random Forest, Linear Regression, XGBoost)
        - Geospatial Data Analysis (Geopy)
        - Model Serving (Joblib)
        """
    )

# --- Main Function ---
def main():
    # Define the sidebar with pages
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ("Home", "Prediction", "Model Details", "About"))

    # Add MLflow dashboard link in sidebar
    mlflow_url = "http://127.0.0.1:5000"
    st.sidebar.markdown("### 📊 MLflow Tracking Dashboard")
    st.sidebar.markdown(f'<a href="{mlflow_url}" target="_blank">Open MLflow UI</a>', unsafe_allow_html=True)

    if page == "Home":
        home_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Model Details":
        model_details_page()
    elif page == "About":
        about_page()

# Run the app
if __name__ == "__main__":
    main()
