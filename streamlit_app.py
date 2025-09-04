import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import requests

# Page configuration
st.set_page_config(
    page_title="Air Quality Prediction API",
    page_icon="🌬️",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_airq_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return model, scaler, True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, False

model, scaler, models_loaded = load_models()

# API Functions
def predict_aqi(temp_c, humidity, gas_score, hour=None, day_of_week=None, month=None, is_weekend=None):
    if not models_loaded:
        return {"error": "Models not available"}
    
    try:
        now = datetime.now()
        
        if hour is None: hour = now.hour
        if day_of_week is None: day_of_week = now.weekday()
        if month is None: month = now.month
        if is_weekend is None: is_weekend = 1 if now.weekday() >= 5 else 0
        
        features = [
            float(temp_c), float(humidity), float(gas_score),
            int(hour), int(day_of_week), int(month), int(is_weekend)
        ]
        
        X = np.array(features).reshape(1, -1)
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
            prediction = float(model.predict(X_scaled)[0])
        else:
            prediction = float(model.predict(X)[0])
        
        prediction = max(0, min(500, prediction))
        confidence = prediction * 0.15
        
        return {
            "prediction": round(prediction, 2),
            "confidence_lower": round(max(0, prediction - confidence), 2),
            "confidence_upper": round(prediction + confidence, 2),
            "timestamp": now.isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

def generate_forecast(district="colombo", hours=24):
    if not models_loaded:
        return {"error": "Models not available"}
    
    try:
        now = datetime.now()
        forecasts = []
        
        variations = {
            'colombo': {'temp_offset': 0, 'humidity_offset': 0, 'gas_offset': 5},
            'kandy': {'temp_offset': -3, 'humidity_offset': 5, 'gas_offset': -2},
            'galle': {'temp_offset': 1, 'humidity_offset': 8, 'gas_offset': -1}
        }
        
        var = variations.get(district.lower(), {'temp_offset': 0, 'humidity_offset': 0, 'gas_offset': 0})
        
        for hour in range(hours):
            future = now + timedelta(hours=hour)
            
            temp_cycle = 4 * np.sin((future.hour - 14) * np.pi / 12)
            gas_cycle = 3 * np.sin((future.hour - 8) * np.pi / 8)
            
            features = [
                28 + var['temp_offset'] + temp_cycle + np.random.normal(0, 1),
                70 + var['humidity_offset'] + np.random.normal(0, 3),
                25 + var['gas_offset'] + gas_cycle + np.random.normal(0, 2),
                future.hour, future.weekday(), future.month,
                1 if future.weekday() >= 5 else 0
            ]
            
            X = np.array(features).reshape(1, -1)
            
            if scaler is not None:
                X_scaled = scaler.transform(X)
                pred = float(model.predict(X_scaled)[0])
            else:
                pred = float(model.predict(X)[0])
            
            pred = max(0, min(300, pred))
            
            forecasts.append({
                "time": future.strftime("%H:%M"),
                "predicted_aqi": round(pred, 1),
                "hour": future.hour
            })
        
        return {"district": district, "forecast": forecasts}
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
st.title("🌬️ Air Quality Prediction API")
st.markdown("Machine Learning-powered air quality predictions for academic research")

# Status indicator
if models_loaded:
    st.success(f"✅ Models loaded successfully: {type(model).__name__}")
else:
    st.error("❌ Models not available")

# API endpoints simulation
tab1, tab2, tab3 = st.tabs(["Single Prediction", "24-Hour Forecast", "API Documentation"])

with tab1:
    st.header("Single Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        temp_c = st.number_input("Temperature (°C)", value=28.0, min_value=-10.0, max_value=50.0)
    with col2:
        humidity = st.number_input("Humidity (%)", value=70.0, min_value=0.0, max_value=100.0)
    with col3:
        gas_score = st.number_input("Gas Score", value=25.0, min_value=0.0, max_value=100.0)
    
    if st.button("Predict AQI"):
        result = predict_aqi(temp_c, humidity, gas_score)
        if "error" in result:
            st.error(result["error"])
        else:
            st.json(result)

with tab2:
    st.header("24-Hour Forecast")
    
    district = st.selectbox("Select District", ["colombo", "kandy", "galle", "jaffna", "ampara"])
    
    if st.button("Generate Forecast"):
        result = generate_forecast(district)
        if "error" in result:
            st.error(result["error"])
        else:
            # Display forecast as chart
            forecast_data = result["forecast"]
            df = pd.DataFrame(forecast_data)
            st.line_chart(data=df.set_index("time")["predicted_aqi"])
            st.json(result)

with tab3:
    st.header("API Documentation")
    st.markdown(f"""
    ### Base URL
    ```
    https://your-app.streamlit.app
    ```
    
    ### Endpoints
    
    **POST /predict**
    ```json
    {{
        "temp_c": 28,
        "humidity": 70,
        "gas_score": 25
    }}
    ```
    
    **GET /forecast/{{district}}**
    
    ### JavaScript Example
    ```javascript
    const API_BASE_URL = 'https://your-app.streamlit.app';
    
    // For Streamlit apps, use the component API
    const response = await fetch(`${{API_BASE_URL}}/_stcore/health`);
    ```
    
    ### Note
    Streamlit apps work differently from traditional APIs. 
    For your frontend integration, consider using:
    1. This Streamlit app for testing
    2. Railway/Render for production API endpoints
    """)