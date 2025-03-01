#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
import io

# --- App Title ---
st.set_page_config(page_title="Renewable Energy Forecast Insight (REFI)", layout="wide")
st.title("☀️ Renewable Energy Forecast Insight (REFI)")
st.markdown("**Analyze, Predict, and Optimize Renewable Energy Forecasts**")

# --- Upload Real-World Dataset ---
st.sidebar.header("Upload Energy Dataset")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# --- WeatherAPI Integration (Using Streamlit Secrets) ---
API_KEY = None
import os

try:
    API_KEY = st.secrets["WEATHER_API_KEY"]
    if API_KEY:
        
    

if API_KEY is None:
    st.error("Weather API key is missing! Add it to environment variables or `.streamlit/secrets.toml` (local) or Streamlit Cloud settings.")
    st.error("Weather API key is missing! Add it to environment variables or `.streamlit/secrets.toml` (local) or Streamlit Cloud settings.")
    API_KEY = ""  # Assign an empty string to prevent further errors
    st.error("Weather API key is missing! Add it to environment variables or `.streamlit/secrets.toml` (local) or Streamlit Cloud settings.")
    API_KEY = ""
    st.error("Weather API key is missing! Add it to environment variables or `.streamlit/secrets.toml` (local) or Streamlit Cloud settings.")
    st.error("Weather API key is missing! Add it to `.streamlit/secrets.toml` (local) or Streamlit Cloud settings.")
    API_KEY = st.secrets["WEATHER_API_KEY"]
else:
    st.error("Weather API key is missing! Add it to .streamlit/secrets.toml or Streamlit Cloud settings.")

BASE_URL = "http://api.weatherapi.com/v1/current.json"

st.sidebar.header("Fetch Real-Time Weather Data")
city = st.sidebar.text_input("Enter City for Weather Forecast", "Oldenburg")

if st.sidebar.button("Get Weather Data") and API_KEY:
    url = f"{BASE_URL}?key={API_KEY}&q={city}&aqi=no"
    response = requests.get(url)

    if response.status_code == 200:
        weather_data = response.json()
        
        # Extract key meteorological variables
        temp_c = weather_data["current"]["temp_c"]
        wind_speed = weather_data["current"]["wind_kph"]
        wind_speed_mps = round(wind_speed / 3.6, 2)
        humidity = weather_data["current"]["humidity"]
        pressure = weather_data["current"]["pressure_mb"]
        cloud_cover = weather_data["current"]["cloud"]
        uv_index = weather_data["current"]["uv"]
        wind_dir = weather_data["current"]["wind_dir"]
        dew_point = weather_data["current"].get("dewpoint_c", "N/A")
        visibility = weather_data["current"].get("vis_km", "N/A")
        wind_gust = weather_data["current"].get("gust_kph", "N/A")
        precipitation = weather_data["current"].get("precip_mm", "N/A")
        air_density = round(pressure / (287.05 * (temp_c + 273.15)), 3)  # Simplified air density calculation

        # Display enhanced weather insights
        st.sidebar.success(
            f"🌍 **City:** {city}  
"
            f"🌡️ **Temperature:** {temp_c}°C  
"
            f"💨 **Wind Speed:** {wind_speed_mps} m/s  
"
            f"💧 **Humidity:** {humidity}%  
"
            f"🔵 **Pressure:** {pressure} hPa  
"
            f"☁️ **Cloud Cover:** {cloud_cover}%  
"
            f"🔆 **UV Index:** {uv_index}  
"
            f"🧭 **Wind Direction:** {wind_dir}  
"
            f"❄️ **Dew Point:** {dew_point}°C  
"
            f"👀 **Visibility:** {visibility} km  
"
            f"💨 **Wind Gusts:** {wind_gust} km/h  
"
            f"🌧️ **Precipitation:** {precipitation} mm  
"
            f"🔬 **Air Density:** {air_density} kg/m³"
        )
    else:
        st.sidebar.error("Invalid city or API issue.")
    url = f"{BASE_URL}?key={API_KEY}&q={city}&aqi=no"
    response = requests.get(url)
    
    
            f"🌍 **City:** {city}  \n"
            f"🌡️ **Temperature:** {temp_c}°C  \n"
            f"💨 **Wind Speed:** {wind_speed_mps} m/s"
        )
    else:
        st.sidebar.error("Invalid city or API issue.")

# --- Load Dataset ---
if file:
    df = pd.read_csv(file, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    st.subheader("📊 Uploaded Data Preview")
    st.write(df.head())

    # --- Separate Visualizations ---
    col1, col2 = st.columns(2)
    with col1:
        fig_solar = px.line(df, x="timestamp", y="solar_power", title="Solar Power Production (timestamp vs. solar_power)")
        st.plotly_chart(fig_solar)
    with col2:
        fig_wind = px.line(df, x="timestamp", y="wind_power", title="Wind Power Production (timestamp vs. wind_power)")
        st.plotly_chart(fig_wind)

    # --- Forecasting Models ---
    forecast_method = st.sidebar.selectbox("Choose Forecasting Model", ["XGBoost", "Random Forest", "Linear Regression", "Support Vector Regression"])
    period = st.sidebar.slider("Forecast Period (days)", 1, 30, 7)

    # Prepare Data for ML Models
    df_ml = df.set_index("timestamp")
    df_ml["target"] = df_ml["solar_power"].shift(-1)
    train = df_ml.dropna()
    X_train, y_train = train.drop(columns=["target"]), train["target"]

    # --- XGBoost Model ---
    if forecast_method == "XGBoost":
        st.subheader("🚀 XGBoost Forecasting Model (Index vs. Predicted Solar Power)")
        model_xgb = xgb.XGBRegressor(objective="reg:squarederror")
        model_xgb.fit(X_train, y_train)
        forecast_xgb = model_xgb.predict(X_train[-period:])
        st.line_chart(forecast_xgb)
    
    # --- Random Forest Model ---
    elif forecast_method == "Random Forest":
        st.subheader("🌳 Random Forest Regressor Forecast (Index vs. Predicted Solar Power)")
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(X_train, y_train)
        forecast_rf = model_rf.predict(X_train[-period:])
        st.line_chart(forecast_rf)
    
    # --- Linear Regression Model ---
    elif forecast_method == "Linear Regression":
        st.subheader("📈 Linear Regression Forecast (Index vs. Predicted Solar Power)")
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        forecast_lr = model_lr.predict(X_train[-period:])
        st.line_chart(forecast_lr)
    
    # --- Support Vector Regression (SVR) Model ---
    elif forecast_method == "Support Vector Regression":
        st.subheader("🔢 Support Vector Regression (SVR) Forecast (Index vs. Predicted Solar Power)")
        model_svr = SVR(kernel="rbf")
        model_svr.fit(X_train, y_train)
        forecast_svr = model_svr.predict(X_train[-period:])
        st.line_chart(forecast_svr)
    
    # --- Forecast Evaluation ---
    st.subheader("🔍 Forecast Accuracy & Error Metrics")
    y_true = df["solar_power"][-period:].values
    y_pred = forecast_xgb if forecast_method == "XGBoost" else forecast_rf if forecast_method == "Random Forest" else forecast_lr if forecast_method == "Linear Regression" else forecast_svr
    
    if len(y_true) == len(y_pred):
        st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_true, y_pred):.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    else:
        st.warning("Not enough data points for error calculation.")

    # --- Generate Reports ---
    st.subheader("📜 Export Forecast Report")
    report_buffer = io.BytesIO()
    report_data = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    report_data.to_csv(report_buffer, index=False)
    st.download_button("Download Report as CSV", report_buffer, file_name="forecast_report.csv", mime="text/csv")
