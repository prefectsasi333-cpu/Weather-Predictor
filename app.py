import os
import json
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ----------------------
# Config / Load artifacts
# ----------------------
MODEL_DIR = "models"

LR_PATH = os.path.join(MODEL_DIR, "lr_model.pkl")
RF_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
XCOLS_PATH = os.path.join(MODEL_DIR, "X_cols.json")
ALPHA_PATH = os.path.join(MODEL_DIR, "hybrid_alpha.json")

# Load models & metadata
lr_model = joblib.load(LR_PATH)
rf_model = joblib.load(RF_PATH)

with open(XCOLS_PATH, "r") as f:
    X_cols = json.load(f)

with open(ALPHA_PATH, "r") as f:
    alpha_obj = json.load(f)

if isinstance(alpha_obj, dict) and "alpha" in alpha_obj:
    best_alpha = float(alpha_obj["alpha"])
else:
    best_alpha = float(alpha_obj)

# API key: prefer environment variable, fallback to placeholder (you can keep hardcoded for quick test)
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Get API key
API_KEY = os.getenv("OWM_KEY")

# ----------------------
# Utility functions
# ----------------------
def comfort_index(temp, humidity, wind):
    temp_score = max(0, 1 - abs(temp - 22) / 30)
    hum_score = max(0, 1 - humidity / 100)
    base = 0.6 * temp_score + 0.3 * hum_score + 0.1 * (1 - min(wind, 20) / 20)
    return int(base * 100)

def categorize_weather(temp, rain_prob):
    if rain_prob > 70: return "🌧 Rainy"
    elif temp > 30: return "☀️ Sunny"
    elif 20 <= temp <= 30: return "🌥 Pleasant"
    elif temp < 10: return "❄ Cold"
    else: return "🌩 Stormy"

def outfit_recommendation(temp, rain_prob, wind):
    if rain_prob > 70: return "Carry an umbrella ☔"
    if temp > 32: return "Wear light cotton clothes 👕"
    if temp < 12: return "Wear warm layers 🧥"
    if wind > 15: return "Avoid cycling today 🚴"
    return "Enjoy your day 🌞"

def hybrid_predict_from_df(df_input):
    # ensure df_input contains columns in X_cols order
    df_aligned = pd.DataFrame([[df_input.get(c, np.nan) for c in X_cols]], columns=X_cols)
    # models expect same column names; the pipelines in the pickles should include imputers
    pred_lr = lr_model.predict(df_aligned)[0]
    pred_rf = rf_model.predict(df_aligned)[0]
    pred_hybrid = best_alpha * pred_lr + (1 - best_alpha) * pred_rf
    return pred_lr, pred_rf, pred_hybrid

def fetch_forecast(city, offset_hours=0):
    """Fetch forecast from OpenWeather 5-day / 3hr API.
       offset_hours = 0 -> first slot (approx now); 24 -> ~24h later (offset index 8).
       returns dict of raw fields and the 'feels_like' for comparison when available.
    """
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    r = requests.get(url, timeout=10)
    data = r.json()
    if "list" not in data:
        raise ValueError(f"API error: {data.get('message','no list')}")
    # compute index (3-hour slots)
    idx = int(offset_hours / 3) if offset_hours >= 0 else 0
    idx = min(idx, len(data["list"]) - 1)
    slot = data["list"][idx]
    return slot

def build_features_from_slot(slot):
    """Build the exact feature dictionary used in training (keys = X_cols).
       Use available slot fields; where not available, use np.nan or computed placeholders.
    """
    main = slot.get("main", {})
    wind = slot.get("wind", {})
    dt = datetime.fromtimestamp(slot.get("dt", datetime.now().timestamp()))
    # Some keys may not be present in API, use fallbacks
    t_max = main.get("temp_max", main.get("feels_like", np.nan))
    t_min = main.get("temp_min", main.get("feels_like", np.nan))
    feels = main.get("feels_like", np.nan)
    hum = main.get("humidity", np.nan)
    pres = main.get("pressure", np.nan)
    wind_spd = wind.get("speed", np.nan)
    rain_3h = slot.get("rain", {}).get("3h", 0.0)
    # Build raw features (as in training)
    features = {
        "feelslikemax": t_max,
        "feelslikemin": t_min,
        "feelslike": feels,
        "humidity": hum,
        "precip": rain_3h,
        "precipprob": 0.0,
        "precipcover": 0.0,
        "windspeed": wind_spd,
        "sealevelpressure": pres,
        "Year": dt.year,
        "month": dt.month,
        "dayofweek": dt.weekday(),
        "dayofyear": dt.timetuple().tm_yday,
        "year-2000": dt.year - 2000,
        "weekofyear": dt.isocalendar()[1],
        "tempmax_humidity": (t_max * hum) if (not pd.isna(t_max) and not pd.isna(hum)) else np.nan,
        "tempmin_humidity": (t_min * hum) if (not pd.isna(t_min) and not pd.isna(hum)) else np.nan,
        "temp_humidity": ((main.get("temp", feels) * hum) if (not pd.isna(hum)) else np.nan),
        "feelslikemax_humidity": (t_max * hum) if (not pd.isna(t_max) and not pd.isna(hum)) else np.nan,
        "feelslikemin_humidity": (t_min * hum) if (not pd.isna(t_min) and not pd.isna(hum)) else np.nan,
        "feelslike_humidity": (feels * hum) if (not pd.isna(feels) and not pd.isna(hum)) else np.nan,
        "temp_range": (t_max - t_min) if (not pd.isna(t_max) and not pd.isna(t_min)) else np.nan,
        "heat_index": (feels + 0.1 * hum) if (not pd.isna(feels) and not pd.isna(hum)) else np.nan
    }
    return features

# ----------------------
# Streamlit UI
# ----------------------
st.title("🌤 Weather Predictor")
st.write("Two modes: manual input, live API forecast.")

mode = st.radio("Select mode:", ["Mode 1: Manual", "Mode 2: Live API"])


# ---------- Mode 1: manual input ----------
if mode == "Mode 1: Manual":
    st.subheader("Mode 2 — Manual Input")
    col1, col2 = st.columns(2)
    with col1:
        feelslike = st.number_input("Feels Like Temp (°C)", value=25.0, format="%.1f")
        humidity = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0, format="%.1f")
        windspeed = st.number_input("Wind Speed (m/s)", value=3.0, format="%.1f")
    with col2:
        tmax = st.number_input("Temp Max (°C)", value=feelslike, format="%.1f")
        tmin = st.number_input("Temp Min (°C)", value=feelslike, format="%.1f")
        pressure = st.number_input("Pressure (hPa)", value=1013.0, format="%.1f")
    if st.button("Predict (Manual)"):
        now = datetime.now()
        slot_like = {
            "main": {"temp_max": tmax, "temp_min": tmin, "feels_like": feelslike, "humidity": humidity, "pressure": pressure},
            "wind": {"speed": windspeed},
            "dt": int(now.timestamp())
        }
        features = build_features_from_slot(slot_like)
        pred_lr, pred_rf, pred_h = hybrid_predict_from_df(features)
        ci = comfort_index(pred_h, humidity, windspeed)
        cat = categorize_weather(pred_h, 0)
        outfit = outfit_recommendation(pred_h, 0, windspeed)
        st.success(f"Hybrid prediction: {pred_h:.1f}°C")
        st.write(f"Comfort Index: {ci}/100")
        st.write(f"Sky Condition: {cat}")
        st.write(f"Recommendation: {outfit}")

# ---------- Mode 2: Live API ----------
else:
    st.subheader("Mode 2 — Live API Prediction")
    city = st.text_input("City name:", value="Chennai")
    day_choice = st.selectbox("Predict for:", ["today (next slot)", "tomorrow (~24h)"])
    offset = 0 if day_choice.startswith("today") else 24

    if st.button("Fetch & Predict (Live)"):
        try:
            slot = fetch_forecast(city, offset_hours=offset)
            features = build_features_from_slot(slot)
            pred_lr, pred_rf, pred_h = hybrid_predict_from_df(features)

            # get humidity and wind for comfort index if present
            hum = slot.get("main", {}).get("humidity", np.nan)
            wind_spd = slot.get("wind", {}).get("speed", np.nan)
            # attempt to read 'feels_like' as API actual
            api_actual = slot.get("main", {}).get("feels_like", np.nan)
            # estimate rain prob if 'pop' exists in the slot (some endpoints include pop)
            rain_prob = int(slot.get("pop", 0) * 100) if "pop" in slot else 0

            ci = comfort_index(pred_h, hum if not pd.isna(hum) else 0, wind_spd if not pd.isna(wind_spd) else 0)
            cat = categorize_weather(pred_h, rain_prob)
            outfit = outfit_recommendation(pred_h, rain_prob, wind_spd if not pd.isna(wind_spd) else 0)

            st.success(f"Hybrid prediction: {pred_h:.1f}°C")
            st.write(f"LR: {pred_lr:.1f}°C    RF: {pred_rf:.1f}°C")
            if not pd.isna(api_actual):
                st.write(f"API feels_like (for reference): {api_actual:.1f}°C")
            st.write(f"Comfort Index: {ci}/100")
            st.write(f"Sky Condition: {cat}")
            st.write(f"Recommendation: {outfit}")

        except Exception as e:
            st.error(f"API / prediction error: {e}")

# Footer
st.markdown("---")
st.caption("Model: hybrid (LR + RF). Make sure models/ contains lr_model.pkl, rf_model.pkl, X_cols.json, hybrid_alpha.json")
