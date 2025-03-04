#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import plotly.express as px

# Streamlit App Title
st.set_page_config(page_title="Wind Power Forecasting System", layout="wide")
st.title("💨 Wind Power Forecasting System")
st.markdown("### ML-Driven Wind Energy Forecasting")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Raw Data Preview")
    st.dataframe(df.head())
    
    # Checking missing values
    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Dataset contains missing values. They will be filled with median values.")
        df.fillna(df.median(), inplace=True)
    
    # Feature and Target Selection
    features = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 'windspeed_10m',
                'windspeed_100m', 'winddirection_10m', 'winddirection_100m', 'windgusts_10m']
    target = 'Power'
    
    X = df[features]
    y = df[target]
    
    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Selection Dropdown
    model_choice = st.selectbox("Select Machine Learning Model", ["Random Forest", "XGBoost"])
    
    if model_choice == "Random Forest":
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20]
        }
    else:
        model = xgb.XGBRegressor()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    
    # Hyperparameter Tuning
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test_scaled)
    
    # Performance Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Display Metrics
    st.write("### 📈 Model Performance Metrics")
    st.write(f"- **Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"- **Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"- **R² Score:** {r2:.4f}")
    
    # Visualizations with Plotly
    st.write("### 🔍 Feature Importance")
    feature_importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    fig = px.bar(feature_importances, x=feature_importances.values, y=feature_importances.index, orientation='h', title="Feature Importance in Wind Power Forecasting", labels={'x': 'Importance Score', 'y': 'Feature'})
    st.plotly_chart(fig)
    
    # Save Model
    joblib.dump(best_model, "wind_forecasting_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    st.success("✅ Best Model trained and saved successfully!")
    
    # User Input for Prediction
    st.write("### 🔮 Make a Prediction")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
    
    if st.button("🚀 Predict Wind Power Generation"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = best_model.predict(input_scaled)[0]
        st.success(f"💨 Predicted Wind Power Generation: {prediction:.2f} kW")
    
    # Forecasting for Multiple Months
    st.write("### 📅 Forecast Wind Power for Different Months")
    forecast_months = st.selectbox("Select forecast period:", [1, 3, 6, 12])
    
    future_dates = pd.date_range(start=datetime.datetime.today(), periods=forecast_months * 30, freq='D').to_pydatetime().tolist()
    future_data = np.tile(X_test_scaled.mean(axis=0), (forecast_months * 30, 1))
    future_predictions = best_model.predict(future_data)
    
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Power Generated (kW)": future_predictions
    })
    
    st.write("### 🔮 Forecast Results")
    st.dataframe(forecast_df)
    
    # Line Chart for Forecast using Plotly
    fig = px.line(forecast_df, x="Date", y="Predicted Power Generated (kW)", markers=True, title=f"Wind Power Forecast for Next {forecast_months} Months")
    st.plotly_chart(fig)

