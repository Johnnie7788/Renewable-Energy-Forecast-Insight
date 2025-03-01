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

API_KEY = st.secrets["weather_api_key"]

if not API_KEY:
    st.error("Weather API key is missing! Add it to environment variables or `.streamlit/secrets.toml` (local) or Streamlit Cloud settings.")

if API_KEY is None:
    st.error("Weather API key is missing! Add it to `.streamlit/secrets.toml` (local) or Streamlit Cloud settings.")
    API_KEY = st.secrets["weather_api_key"]
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
        temp_c = weather_data["current"]["temp_c"]
        wind_speed = weather_data["current"]["wind_kph"]  # Convert from km/h to m/s
        wind_speed_mps = round(wind_speed / 3.6, 2)

        st.sidebar.success(
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

